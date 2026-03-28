import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ConfiguracaoModelo:
    dimensao_modelo: int = 512
    num_cabecas: int = 8
    num_camadas: int = 6
    dimensao_ff: int = 2048
    dropout: float = 0.1
    profundidade_estocastica: float = 0.2
    num_experts: int = 8
    capacidade_expert: float = 1.0
    vocab_tamanho: int = 32000
    sequencia_max_len: int = 512
    imagem_tamanho: int = 224
    canais_imagem: int = 3
    num_features_tabulares: int = 20

@dataclass
class ConfiguracaoTreino:
    lote_tamanho: int = 32
    acumulacao_gradientes: int = 4
    taxa_aprendizado: float = 1e-4
    peso_decaimento: float = 0.05
    epocas: int = 100
    warmup_passos: int = 2000
    clipping_gradiente: float = 1.0
    precisao_mista: bool = True
    caminho_checkpoints: str = "checkpoints/"
    projeto_wandb: str = "projeto-multimodal-elite"
    dispositivos: List[int] = field(default_factory=lambda: [0])

def obter_configuracao_padrao() -> ConfiguracaoModelo:
    return ConfiguracaoModelo()
