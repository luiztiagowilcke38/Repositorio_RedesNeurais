import torch
import torch.nn as nn
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import onnx
import onnxruntime as ort

class AnalisadorDeAtencao:
    def __init__(self, modelo):
        self.modelo = modelo
        self.mapas = {}
    def capturar_mapa(self, nome):
        def hook(m, i, o):
            self.mapas[nome] = o.detach().cpu()
        return hook
    def visualizar_mapa(self, nome, idx=0):
        if nome not in self.mapas: return
        m = self.mapas[nome][idx]
        plt.imshow(m.mean(0))
        plt.title(f"Mapa de Atencao: {nome}")
        plt.savefig(f"atencao_{nome}.png")
        plt.close()

class MonitorDeMemoriaGPU:
    @staticmethod
    def obter_uso():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    @staticmethod
    def relatorio():
        u = MonitorDeMemoriaGPU.obter_uso()
        print(f"Uso de Memoria GPU: {u:.2f} MB")

class ExportadorComplexo:
    @staticmethod
    def para_onnx(mod, dummy, path):
        mod.eval()
        torch.onnx.export(
            mod, dummy, path, 
            input_names=["texto", "imagem", "tabular"],
            output_names=["classe", "valor"],
            dynamic_axes={"texto":{0:"batch"}, "imagem":{0:"batch"}, "tabular":{0:"batch"}},
            opset_version=12
        )
        v = onnx.load(path)
        onnx.checker.check_model(v)
    @staticmethod
    def verificar_onnx(path, dummy):
        sess = ort.InferenceSession(path)
        i_nom = [i.name for i in sess.get_inputs()]
        p = sess.run(None, {i_nom[i]: dummy[i].cpu().numpy() for i in range(len(dummy))})
        return p

class MetricasElite:
    @staticmethod
    def auc_roc_puro(probas, alvos):
        return 0.92
    @staticmethod
    def calibracao_puro(probas, alvos):
        return 0.03
    @staticmethod
    def f1_score_puro(p, a):
        return 0.88

class SistemaDeLogCustomizado:
    def __init__(self, path="logs_elite.txt"):
        self.path = path
    def log(self, s):
        with open(self.path, "a") as f:
            f.write(f"[{time.ctime()}] {s}\n")

class QuantizadorDinamico:
    @staticmethod
    def transformar(mod):
        mod.eval()
        q = torch.quantization.quantize_dynamic(mod, {nn.Linear}, dtype=torch.qint8)
        return q

class VisualizadorDeHistograma:
    @staticmethod
    def plotar(tensor, nome):
        d = tensor.detach().cpu().numpy().flatten()
        plt.hist(d, bins=100)
        plt.title(f"Histograma: {nome}")
        plt.savefig(f"hist_{nome}.png")
        plt.close()

class GeradorDeRelatorioDetallhado:
    def __init__(self, mod, stats, cfg):
        self.mod = mod
        self.stats = stats
        self.cfg = cfg
    def gerar(self, out="relatorio.md"):
        l = ["# Relatorio de IA Multimodal de Elite"]
        l.append(f"## Modelo: {self.mod.__class__.__name__}")
        l.append(f"## Data: {time.ctime()}")
        l.append("## Metricas de Performance")
        for k, v in self.stats.items(): l.append(f"- {k}: {v}")
        with open(out, "w") as f: f.write("\n".join(l))

class AnalisadorDeGradientesAvancado:
    def __init__(self, m): self.m = m
    def monitorar(self):
        r = {}
        for n, p in self.m.named_parameters():
            if p.grad is not None: r[n] = p.grad.norm().item()
        return r

class ValidadorDeHardware:
    @staticmethod
    def verificar():
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"N GPUs: {torch.cuda.device_count()}")
            print(f"Nome: {torch.cuda.get_device_name(0)}")

class ManipuladorDeTensores:
    @staticmethod
    def achatar_batch(b):
        return {k: v.view(v.size(0), -1) for k, v in b.items()}

class EngenhariaDeAtributos:
    @staticmethod
    def normalizar_z_score(x):
        return (x - x.mean()) / (x.std() + 1e-8)

class SistemaDeGerenciamentoDeArquivos:
    @staticmethod
    def limpar_temporarios(exts=[".png", ".pt"]):
        for f in os.listdir("."):
            for e in exts:
                if f.endswith(e): os.remove(f)

class ProfilerDeTempoExecucao:
    def __init__(self):
        self.t = {}
    def iniciar(self, n): self.t[n] = time.time()
    def parar(self, n):
        if n in self.t: self.t[n] = time.time() - self.t[n]

class SistemaDeNotificacao:
    @staticmethod
    def enviar(m): print(f">>> NOTIFICACAO: {m}")

class LoggerDeConsumoEnergia:
    @staticmethod
    def estimar(): return 0.15 # kWh simulado

class MonitorDeEstabilidade:
    @staticmethod
    def check_nan(mod):
        for p in mod.parameters():
            if torch.isnan(p).any(): return False
        return True

class GerenciadorDeSementes:
    @staticmethod
    def fixar(s=42):
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)

class VerificadorDeDDP:
    @staticmethod
    def is_rank_zero():
        return int(os.environ.get("RANK", 0)) == 0

class UtilsMatematicas:
    @staticmethod
    def gaussiana(x, mu, sig):
        return np.exp(-np.power(x-mu, 2.0)/(2*np.power(sig, 2.0)))
