import torch

class HiperparametrosAvancados:
    DIM_MODELO_PADRAO = 1024
    NUM_CABECAS_PADRAO = 16
    NUM_CAMADAS_ENCODER = 12
    RAZAO_EXPANSAO_FF = 4
    DROPOUT_RESIDUAL = 0.15
    STOCHASTIC_DEPTH_PROB = 0.25
    
    # Configuracoes de Mixture of Experts
    MOE_NUM_EXPERTS = 16
    MOE_TOP_K = 2
    MOE_CAPACIDADE_FATOR = 1.25
    MOE_BIASED_GATING = False
    
    # Configuracoes de Visao
    VISION_PATCH_SIZE = 16
    VISION_STRIDE = 8
    VISION_CHANNELS = [64, 128, 256, 512, 1024]
    VISION_USE_SE = True
    
    # Configuracoes de Texto (RoPE)
    ROPE_THETA = 10000.0
    ROPE_SCALING_FACTOR = 1.0
    
    # Configuracoes de Otimizacao
    OPTIM_EPS = 1e-8
    OPTIM_BETAS = (0.9, 0.999)
    OPTIM_WEIGHT_DECAY = 0.01
    OPTIM_LR_WARMUP_STEPS = 5000
    OPTIM_LR_MIN = 1e-7

class PadroesDeInicializacao:
    @staticmethod
    def aplicar(modelo):
        for nome, parametro in modelo.named_parameters():
            if 'weight' in nome:
                if len(parametro.shape) >= 2:
                    torch.nn.init.orthogonal_(parametro)
                else:
                    torch.nn.init.constant_(parametro, 1.0)
            elif 'bias' in nome:
                torch.nn.init.constant_(parametro, 0.0)

class ConstantesDeProducao:
    VERSAO_API = "v1.0.0-elite"
    NOME_PROJETO = "MultimodalElite_LuizTiago"
    FORMATO_CHECKPOINT = ".pt"
    FORMATO_ONNX = ".onnx"
    QUALIDADE_QUANTIZACAO = "INT8"

class LimitesOperacionais:
    MAX_TEXTO_LEN = 1024
    MAX_IMAGEM_RES = 1024
    MAX_BATCH_SIZE_GPU = 128
    MAX_GPU_MEMORY_GB = 40.0

class MetadadosArquitetura:
    ESTADO_ARTE = True
    COMPLEXIDADE = "Ultra"
    MODALIDADES = ["Vision", "Text", "Tabular"]
    COMPONENTES = ["Transformer", "CNN", "RNN", "MoE"]

class EsquemaDeCoresVisualizacao:
    ATENCAO_MAPA = "magma"
    FEATURE_MAP = "viridis"
    PERDA_GRAFICO = "coral"
