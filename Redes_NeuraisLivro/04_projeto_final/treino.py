import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
import os
from typing import Dict, List, Any
import wandb

class PipelineTreinamentoAvancado:
    def __init__(self, modelo, config_treino, config_mod, car_treino, car_val):
        self.modelo = modelo
        self.cfg_t = config_treino
        self.cfg_m = config_mod
        self.car_t = car_treino
        self.car_v = car_val
        
        self.otimizador = optim.AdamW(
            modelo.parameters(), 
            lr=self.cfg_t.taxa_aprendizado, 
            weight_decay=self.cfg_t.peso_decaimento
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.otimizador, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        self.scaler = GradScaler(enabled=self.cfg_t.precisao_mista)
        self.crit_cls = nn.CrossEntropyLoss()
        self.crit_reg = nn.MSELoss()
        
        self.melhor_perda = float("inf")
        self.paciencia_atual = 0

    def treinar_lote(self, b):
        self.otimizador.zero_grad()
        
        with autocast(enabled=self.cfg_t.precisao_mista):
            cls_out, reg_out = self.modelo(
                b["texto"].cuda(), 
                b["imagem"].cuda(), 
                b["tabular"].cuda()
            )
            perda_cls = self.crit_cls(cls_out, b["classe"].cuda())
            perda_reg = self.crit_reg(reg_out, b["valor"].cuda())
            perda_total = perda_cls + 0.5 * perda_reg
            
        if self.cfg_t.precisao_mista:
            self.scaler.scale(perda_total).backward()
            self.scaler.unscale_(self.otimizador)
            nn.utils.clip_grad_norm_(self.modelo.parameters(), self.cfg_t.clipping_gradiente)
            self.scaler.step(self.otimizador)
            self.scaler.update()
        else:
            perda_total.backward()
            nn.utils.clip_grad_norm_(self.modelo.parameters(), self.cfg_t.clipping_gradiente)
            self.otimizador.step()
            
        return perda_total.item()

    def validar(self):
        self.modelo.eval()
        perda_v = 0
        with torch.no_grad():
            for b in self.car_v:
                c, r = self.modelo(b["texto"].cuda(), b["imagem"].cuda(), b["tabular"].cuda())
                perda_v += (self.crit_cls(c, b["classe"].cuda()) + self.crit_reg(r, b["valor"].cuda())).item()
        return perda_v / len(self.car_v)

    def executar(self):
        wandb.init(project=self.cfg_t.projeto_wandb)
        for e in range(self.cfg_t.epocas):
            print(f"Epoca {e}")
            self.modelo.train()
            soma_p = 0
            for i, b in enumerate(self.car_t):
                p = self.treinar_lote(b)
                soma_p += p
                self.scheduler.step(e + i / len(self.car_t))
                
                if i % 10 == 0:
                    wandb.log({"perda_treino": p, "lr": self.scheduler.get_last_lr()[0]})
            
            p_val = self.validar()
            wandb.log({"perda_val": p_val})
            
            if p_val < self.melhor_perda:
                self.melhor_perda = p_val
                self.paciencia_atual = 0
                self.salvar_melhor(e)
            else:
                self.paciencia_atual += 1
                
            if self.paciencia_atual >= 15:
                print("Early stopping ativado")
                break

    def salvar_melhor(self, e):
        if not os.path.exists(self.cfg_t.caminho_checkpoints):
            os.makedirs(self.cfg_t.caminho_checkpoints)
        torch.save(self.modelo.state_dict(), f"{self.cfg_t.caminho_checkpoints}/melhor_modelo.pt")

class GerenciadorDeCheckpoints:
    @staticmethod
    def carregar(modelo, otimizador, caminho):
        cp = torch.load(caminho)
        modelo.load_state_dict(cp["modelo"])
        otimizador.load_state_dict(cp["otim"])
        return cp["epoca"]
