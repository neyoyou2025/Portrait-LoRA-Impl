from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from torchvision.transforms.functional import normalize, resize
from src.utils import img2tensor, tensor2img
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from PIL import Image
import numpy as np
import insightface
import torch
import cv2
import gc

class FaceEncoder:
    def __init__(self, version: str, device, weight_dtype, *args, **kwargs):
        super().__init__()
        self.device = device

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
            model_rootpath="./models"
        )
        self.weight_dtype = weight_dtype
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # antelopev2
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        self.app = FaceAnalysis(
            name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)

        self.processor = CLIPProcessor.from_pretrained(version)
        self.clip_vision_model = CLIPModel.from_pretrained(version)
        self.clip_vision_model.to(self.device, dtype=weight_dtype)
        self.clip_vision_model.eval()
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False

        gc.collect()
        torch.cuda.empty_cache()

        # other configs
        self.debug_img_list = []

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    def get_id_embedding(self, image_path):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        import os
        if os.path.isfile(image_path):
            images = [image_path]
        elif os.path.isdir(image_path):
            images = sorted(os.listdir(image_path))
        else:
            raise ValueError("path error")  
        id_ante_embeddings = []
        image_embeds = []
        for img in images:
            img_path = os.path.join(image_path, img)
            self.face_helper.clean_all()
            image = cv2.imread(img_path)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # get antelopev2 embedding
            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
                    -1
                ]  # only use the maximum face
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(
                    image[
                        int(face_info['bbox'][1]) : int(face_info['bbox'][3]),
                        int(face_info['bbox'][0]) : int(face_info['bbox'][2]),
                    ]
                )
            else:
                id_ante_embedding = None

            # using facexlib to detect and align face
            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) == 0:
                raise RuntimeError('facexlib align face fail')
            align_face = self.face_helper.cropped_faces[0]
            # incase insightface didn't detect face
            if id_ante_embedding is None:
                print('fail to detect face using insightface, extract embedding on align face')
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            # parsing
            input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input = input.to(self.device)
            parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input)
            # only keep the face features
            face_features_image = torch.where(bg, white_image, self.to_gray(input))
            face_features_image = tensor2img(face_features_image, rgb2bgr=False)
            self.debug_img_list.append(face_features_image)

            if face_features_image is None:
                raise "input_img is None"
            
            img = face_features_image.astype(np.uint8)
            # clip img encoder encode portrait
            image = Image.fromarray(img)
            img = self.processor(text=["a"], images=image, return_tensors="pt", padding=True)
            image_embed = self.clip_vision_model(**img.to(self.device)).image_embeds
            
            id_ante_embeddings.append(id_ante_embedding.to(dtype=self.weight_dtype))
            image_embeds.append(image_embed)
        return id_ante_embeddings, image_embeds