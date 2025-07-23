from torch import nn
import torch


from models.diffusion_classifier.classification_head import Resnet3DHead, ResnetFusion, AttentionFusion




class DiffusionVideoClassifier(nn.Module):
    def __init__(self, 
                 num_frames,
                 diffusion_2d_model,
                 classifier_head,
                 sex_embedding: int = 0,
                 age_embedding: int = 0,
                 ppt_embedding: int = 0,
                 num_ppt: int = 3,
                 n_classes = 1):

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_2d_model = diffusion_2d_model
        self.classifier_head = classifier_head
        
        self.num_frame = num_frames

        self.use_sex = sex_embedding != 0
        self.use_age = age_embedding != 0
        self.use_ppt = ppt_embedding != 0

        last_dim = self.classifier_head.hidden_dim
        if self.use_sex:
            self.sex_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=sex_embedding)
            last_dim += sex_embedding
        if self.use_age:
            self.age_embedding_layer = nn.Sequential(nn.Linear(1, age_embedding, bias=False),nn.ReLU())
            last_dim += age_embedding
        if self.use_ppt:
            self.ppt_embedding_layer = nn.Sequential(nn.Linear(num_ppt, ppt_embedding, bias=False),nn.ReLU())
            last_dim += ppt_embedding
        
        self.last = nn.Linear(last_dim, n_classes)


    def forward(self, data):
        clip = data['clip'].to(self.device).float() # (B, C, T, H, W)
        b, c, f, h ,w = clip.shape
        assert f == self.num_frame

        features_2d = []

        for i in range(f):
            features_2d.append(self.diffusion_2d_model(clip[:,:,i,:,:]))
        features_2d = torch.stack(features_2d, dim=2) # (B, C, T, H, W)
        features = self.classifier_head(features_2d)

        if self.use_sex:
            sex = data['sex'].to(self.device).long()
            sex_feature = self.sex_embedding_layer(sex)
            features = torch.cat((features, sex_feature), dim=1)
        
        if self.use_age:
            age = data['age'].to(self.device).unsqueeze(1).float()
            age_feature = self.age_embedding_layer(age)
            features = torch.cat((features, age_feature), dim=1)
        
        if self.use_ppt:
            ppt = data['ppt'].to(self.device).float()
            ppt_feature = self.ppt_embedding_layer(ppt)
            features = torch.cat((features, ppt_feature), dim=1)


        out = self.last(features)
        return out
    

