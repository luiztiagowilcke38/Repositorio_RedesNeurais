import torch
import unittest
import numpy as np
from modelo import ArquiteturaHibridaElite, AtencaoSelfMultiCabeca, InvertedResidual, MixtureExpertsAvancado
from configuracao import obter_configuracao_padrao
from dataset import DatasetMultimodalComplexo, DataLoaderAssincrono
from utilitarios import MonitorDeMemoriaGPU, MetricasElite

class TesteArquiteturaMultimodal(unittest.TestCase):
    def setUp(self):
        self.cfg = obter_configuracao_padrao()
        self.mod = ArquiteturaHibridaElite(self.cfg)
        
    def test_forward_shape(self):
        t = torch.randint(0, self.cfg.vocab_tamanho, (1, self.cfg.sequencia_max_len))
        img = torch.randn(1, 3, 224, 224)
        tab = torch.randn(1, self.cfg.num_features_tabulares)
        saida = self.mod(t, img, tab)
        self.assertEqual(saida.shape, (1, 2))

    def test_moe_layers(self):
        x = torch.randn(2, 512, self.cfg.dimensao_modelo)
        moe = MixtureExpertsAvancado(self.cfg.dimensao_modelo)
        out = moe(x)
        self.assertEqual(out.shape, x.shape)

class TestePipelineDados(unittest.TestCase):
    def test_dataset_item(self):
        ds = DatasetMultimodalComplexo(n_amostras=10)
        item = ds[0]
        self.assertIn("texto", item)
        self.assertIn("imagem", item)
        self.assertEqual(item["imagem"].shape, (3, 224, 224))

    def test_dataloader_batch(self):
        ds = DatasetMultimodalComplexo(n_amostras=50)
        dl = DataLoaderAssincrono(ds, lote=8)
        b = next(iter(dl))
        self.assertEqual(b["texto"].shape[0], 8)

class TesteUtilitarios(unittest.TestCase):
    def test_memoria(self):
        u = MonitorDeMemoriaGPU.obter_uso()
        self.assertIsInstance(u, float)

    def test_metricas(self):
        p = torch.randn(10, 2)
        a = torch.randint(0, 2, (10,))
        s = MetricasElite.auc_roc_puro(p, a)
        self.assertGreater(s, 0.0)

class TesteEstabilidadeNumerica(unittest.TestCase):
    def test_nan_loss(self):
        mod = ArquiteturaHibridaElite(obter_configuracao_padrao())
        t = torch.randint(0, 100, (2, 512))
        img = torch.randn(2, 3, 224, 224)
        tab = torch.randn(2, 20)
        out = mod(t, img, tab)
        loss = out.sum()
        loss.backward()
        for p in mod.parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any())

class TesteStress(unittest.TestCase):
    def test_grande_lote(self):
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        t = torch.randint(0, 100, (16, 512))
        i = torch.randn(16, 3, 224, 224)
        tab = torch.randn(16, 20)
        out = mod(t, i, tab)
        self.assertEqual(out.shape[0], 16)

class TestePersistencia(unittest.TestCase):
    def test_salvamento(self):
        import os
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        torch.save(mod.state_dict(), "temp.pt")
        self.assertTrue(os.path.exists("temp.pt"))
        os.remove("temp.pt")

class TesteDiferenciabilidade(unittest.TestCase):
    def test_gradientes_moe(self):
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        t = torch.randint(0, 100, (1, 512))
        i = torch.randn(1, 3, 224, 224)
        tab = torch.randn(1, 20)
        out = mod(t, i, tab)
        loss = out.sum()
        loss.backward()
        for n, p in mod.named_parameters():
            if "exps" in n:
                self.assertIsNotNone(p.grad)

class TesteAugmentacao(unittest.TestCase):
    def test_transformacoes_aleatorias(self):
        from dataset import AugmentacaoDeImagemAvancada
        aug = AugmentacaoDeImagemAvancada()
        img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        out = aug.processar(img)
        self.assertEqual(out.shape, (3, 224, 224))

class TesteFusaoHibrida(unittest.TestCase):
    def test_cross_modal_weights(self):
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        pesos = mod.fusion.norm.weight
        self.assertEqual(pesos.shape[0], cfg.dimensao_modelo)

class TesteValidacaoFisica(unittest.TestCase):
    def test_batch_norm_running(self):
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        mod.train()
        i = torch.randn(2, 3, 224, 224)
        _ = mod.vision(i)
        self.assertTrue(mod.vision.net[1].running_mean.sum() != 0)

class TesteRecursividade(unittest.TestCase):
    def test_lstm_unroll(self):
        cfg = obter_configuracao_padrao()
        mod = ArquiteturaHibridaElite(cfg)
        tab = torch.randn(4, 20)
        out = mod.tabular(tab)
        self.assertEqual(out.shape, (4, cfg.dimensao_modelo))

if __name__ == "__main__":
    unittest.main()
