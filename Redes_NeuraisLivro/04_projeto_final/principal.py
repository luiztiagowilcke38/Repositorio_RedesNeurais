import argparse
import torch
import os
import sys
import time
from configuracao import obter_configuracao_padrao, ConfiguracaoTreino
from dataset import DatasetMultimodalComplexo, DataLoaderAssincrono
from modelo import FinalEliteModel, CustomModuleRefiner
from treino import PipelineTreinamentoAvancado
from utilitarios import MonitorDeMemoriaGPU, ExportadorComplexo, QuantizadorDinamico, ValidadorDeHardware, GeradorDeRelatorioDetallhado

def exibir_banner_elite():
    print("="*80)
    print("SISTEMA MULTIMODAL DE ELITE - VERSAO MASTERCLASS")
    print("AUTOR: LUIZ TIAGO WILCKE")
    print("ESTADO: ENGENHARIA DE SOFTWARE DE ALTA PERFORMANCE")
    print("="*80)

def exibir_ajuda_detalhada():
    print("MODOS DE OPERACAO:")
    print("  treinar  : Inicia o ciclo completo de treinamento com AMP e SGDR.")
    print("  validar  : Carrega um checkpoint e executa metricas no set de validacao.")
    print("  exportar : Gera arquivo ONNX para deploy em producao (Windows/Linux/Web).")
    print("  quantizar: Converte pesos para INT8 visando dispositivos de borda (Edge).")
    print("  testar   : Executa a suite completa de testes unitarios e integracao.")

def simular_carregamento_producao():
    for i in range(5):
        print(f"Carregando modulos de hardware [{i+1}/5]...", end="\r")
        time.sleep(0.2)
    print("\nAmbiente pronto.")

def orquestrar_elite():
    parser = argparse.ArgumentParser(description="Projeto Multimodal Elite")
    parser.add_argument("--acao", type=str, required=True, choices=["treinar", "validar", "exportar", "quantizar", "testar"])
    parser.add_argument("--lote", type=int, default=32)
    parser.add_argument("--epocas", type=int, default=100)
    parser.add_argument("--taxa", type=float, default=1e-4)
    parser.add_argument("--precisao_mista", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--caminho_modelo", type=str, default="checkpoints/melhor_modelo.pt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    exibir_banner_elite()
    simular_carregamento_producao()
    ValidadorDeHardware.verificar()

    conf_m = obter_configuracao_padrao()
    conf_t = ConfiguracaoTreino(
        lote_tamanho=args.lote, 
        epocas=args.epocas, 
        taxa_aprendizado=args.taxa,
        precisao_mista=args.precisao_mista
    )

    print("Instanciando Arquitetura Hibrida Transformer-CNN-RNN...")
    modelo = FinalEliteModel(conf_m)
    refiner = CustomModuleRefiner(modelo)
    refiner.reset_params()
    
    if torch.cuda.is_available():
        modelo = modelo.cuda()
        print(f"Modelo alocado na GPU com sucesso.")
    
    print("Configurando Pipeline de Dados...")
    ds_t = DatasetMultimodalComplexo(n_amostras=5000)
    ds_v = DatasetMultimodalComplexo(n_amostras=1000)
    dl_t = DataLoaderAssincrono(ds_t, args.lote)
    dl_v = DataLoaderAssincrono(ds_v, args.lote)

    if args.acao == "treinar":
        print("Protocolo de Treinamento Iniciado.")
        treinador = PipelineTreinamentoAvancado(modelo, conf_t, conf_m, dl_t, dl_v)
        treinador.executar()
        print("Fim do Treinamento.")
        
    elif args.acao == "validar":
        print(f"Carregando pesos de: {args.caminho_modelo}")
        modelo.load_state_dict(torch.load(args.caminho_modelo))
        treinador = PipelineTreinamentoAvancado(modelo, conf_t, conf_m, dl_t, dl_v)
        score = treinador.validar()
        print(f"Resultado da Validacao (Loss): {score:.6f}")
        
    elif args.acao == "exportar":
        print("Preparando exportacao ONNX...")
        dummy = (
            torch.randint(0, 100, (1, 1024)).cuda(),
            torch.randn(1, 3, 224, 224).cuda(),
            torch.randn(1, 20).cuda()
        )
        ExportadorComplexo.para_onnx(modelo, dummy, "masterclass_elite.onnx")
        print("Arquivo 'masterclass_elite.onnx' pronto para distribuicao.")
        
    elif args.acao == "quantizar":
        print("Iniciando Quantizacao Dinamica Post-Training...")
        mod_q = QuantizadorDinamico.transformar(modelo.cpu())
        torch.save(mod_q.state_dict(), "elite_quantizado_int8.pt")
        print("Compressao finalizada: elite_quantizado_int8.pt")
        
    elif args.acao == "testar":
        print("Acionando Suite de Testes do Sistema...")
        import subprocess
        res = subprocess.run(["python3", "testes.py"])
        if res.returncode == 0:
            print("Todos os testes passaram com sucesso.")
        else:
            print("Falha detectada nos testes unitarios.")

    MonitorDeMemoriaGPU.relatorio()
    rel = GeradorDeRelatorioDetallhado(modelo, {"status": "concluido", "finalizado": True}, conf_m)
    rel.gerar()
    print("Processo encerrado.")

if __name__ == "__main__":
    try:
        orquestrar_elite()
    except Exception as e:
        print(f"Erro fatal: {e}")
        sys.exit(1)
