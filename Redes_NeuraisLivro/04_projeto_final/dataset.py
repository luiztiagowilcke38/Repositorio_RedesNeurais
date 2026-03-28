import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
from typing import List, Tuple, Dict, Optional

class TokenizadorBPEProfissional:
    def __init__(self, vocab_tamanho: int = 32000):
        self.vocab_tamanho = vocab_tamanho
        self.vocabulario = {chr(i): i for i in range(256)}
        self.reverso = {v: k for k, v in self.vocabulario.items()}
        self.merges = {}

    def treinar(self, textos: List[str]):
        counts = {}
        for t in textos:
            for c in t: counts[c] = counts.get(c, 0) + 1
        
        idx = 256
        while len(self.vocabulario) < self.vocab_tamanho and idx < self.vocab_tamanho:
            par_comum = ("e", "r") # Simulação
            self.vocabulario["".join(par_comum)] = idx
            self.reverso[idx] = "".join(par_comum)
            idx += 1

    def codificar(self, texto: str, max_len: int = 512) -> torch.Tensor:
        tokens = [self.vocabulario.get(c, 0) for c in texto]
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

class AugmentacaoDeImagemAvancada:
    def __init__(self, tamanho: int = 224):
        self.tamanho = tamanho

    def girar(self, img):
        if random.random() > 0.5: return cv2.flip(img, 1)
        return img

    def rotacionar(self, img):
        if random.random() > 0.3:
            ang = random.uniform(-20, 20)
            m = cv2.getRotationMatrix2D((self.tamanho/2, self.tamanho/2), ang, 1)
            return cv2.warpAffine(img, m, (self.tamanho, self.tamanho))
        return img

    def brilho(self, img):
        if random.random() > 0.4:
            fator = random.uniform(0.7, 1.3)
            return np.clip(img * fator, 0, 255).astype(np.uint8)
        return img

    def ruido(self, img):
        if random.random() > 0.2:
            std = random.uniform(5, 15)
            n = np.random.normal(0, std, img.shape).astype(np.uint8)
            return cv2.add(img, n)
        return img

    def processar(self, img):
        img = cv2.resize(img, (self.tamanho, self.tamanho))
        img = self.girar(img)
        img = self.rotacionar(img)
        img = self.brilho(img)
        img = self.ruido(img)
        return torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

class GeradorDeDadosTabulares:
    @staticmethod
    def gerar(n_features: int) -> torch.Tensor:
        base = np.random.randn(n_features).astype(np.float32)
        if random.random() > 0.1:
            base[random.randint(0, n_features-1)] = np.nan
        # Imputação simples
        base = np.nan_to_num(base, nan=0.0)
        return torch.from_numpy(base)

class DatasetMultimodalComplexo(Dataset):
    def __init__(self, n_amostras: int = 2000):
        self.n = n_amostras
        self.token = TokenizadorBPEProfissional()
        self.aug = AugmentacaoDeImagemAvancada()

    def __len__(self): return self.n

    def __getitem__(self, i):
        txt = f"Amostra multimodal numero {i} para treinamento profundo"
        tokens = self.token.codificar(txt)
        
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        img_t = self.aug.processar(img)
        
        tab = GeradorDeDadosTabulares.gerar(20)
        
        label = 1 if i % 3 == 0 else 0
        valor = float(i) / self.n
        
        return {
            "texto": tokens,
            "imagem": img_t,
            "tabular": tab,
            "classe": torch.tensor(label, dtype=torch.long),
            "valor": torch.tensor([valor], dtype=torch.float32)
        }

class DataLoaderAssincrono:
    def __init__(self, ds, lote: int):
        self.dl = DataLoader(
            ds, 
            batch_size=lote, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
    def __iter__(self): return iter(self.dl)
    def __len__(self): return len(self.dl)

class MixUpEstrategia:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    def mix(self, img, lbl):
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(img.size(0))
        mixed_img = l * img + (1 - l) * img[idx]
        return mixed_img, lbl, lbl[idx], l

class CutMixEstrategia:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    def mix(self, img, lbl):
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(img.size(0))
        # Lógica de recorte bbx/bby...
        return img, lbl, lbl[idx], l
