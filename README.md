LCWHVINet: Low-Light Image Enhancement com HVI, Wavelet e Restormer

1. Descrição

A LCWHVINet é uma arquitetura de restauração de imagens de baixa iluminação desenvolvida para reduzir instabilidades de luminância e, principalmente, reconstruções cromáticas excessivamente saturadas. A proposta substitui a reconstrução RGB livre da versão anterior por um processamento desacoplado de intensidade e cromaticidade.

A imagem de entrada é representada no espaço HVI, no qual os canais H e V carregam informação cromática e o canal I representa intensidade. A informação de frequência extraída por Wavelet atua exclusivamente no ramo de intensidade. Os ramos de cor e intensidade são processados por blocos do tipo Restormer e interagem por uma fusão residual controlada. A saída de intensidade é produzida por uma curva limitada, enquanto a cor pode permanecer bloqueada ou receber apenas uma pequena correção limitada.

O pipeline completo trabalha em RGB normalizado no intervalo [0,1].

2. Objetivos do redesign

Os objetivos principais são:

impedir reconstruções RGB independentes e sem restrição;

separar explicitamente correção de iluminação e correção de cor;

permitir um treinamento inicial com cromaticidade bloqueada;

limitar matematicamente a magnitude da correção cromática;

manter Wavelet como fonte de informação de frequência sem reconstruir diretamente RGB;

remover a dependência de janelas espaciais do Swin;

monitorar PSNR, SSIM, LPIPS, erro de intensidade, erro cromático e clipping;

utilizar um único código para LSD/PAMAZONIA, LOL-v1 e LOL-v2.

3. Estrutura da arquitetura

Fluxo simplificado:

RGB [0,1]
   |
   v
RGB -> HVI
   |
   +----------------------+
   |                      |
   v                      v
H,V                    Intensidade I
Cor                    Iluminação
   |                      |
Embedding              Embedding
   |                      |
Restormer              Wavelet
   |                      |
   |                   Restormer
   |                      |
   +---- Cross Fusion ----+
   |                      |
   v                      v
Delta H,V limitado     Curva de intensidade
   |                      |
   v                      v
H,V corrigidos         I corrigida
   +----------+-----------+
              |
              v
             HVI
              |
              v
          HVI -> RGB
              |
              v
          saída [0,1]

No modo color_mode=lock, a correção cromática é exatamente zero:

HV_out = HV_input

No modo color_mode=bounded, a correção é limitada:

delta_HV = color_scale * tanh(raw_delta_HV)
HV_out = HV_input + delta_HV

O valor padrão é:

color_scale = 0.03

A intensidade é modificada por uma curva limitada e diferenciável. As cabeças responsáveis pela curva e pela correção cromática são inicializadas em zero, fazendo com que a arquitetura comece aproximadamente como uma transformação identidade.

4. Arquivos necessários

A estrutura recomendada do projeto é:

/home/unicornio/User/LCWNet/
|
|-- models/
|   |-- __init__.py
|   |-- lcw_hvi_backbone.py
|   `-- loss_hvi.py
|
|-- dataload/
|   |-- __init__.py
|   `-- llie_dataset.py
|
|-- train_lcw_hvi.py
`-- infer_lcw_hvi.py

Os scripts de execução podem ser mantidos em:

/home/unicornio/User/gpuFarm/
|
|-- train_lcw_hvi.sh
|-- infer_lcw_hvi.sh
|-- train_lcw_hvi.slurm
`-- infer_lcw_hvi.slurm

5. Dependências

O código utiliza:

Python
PyTorch
Torchvision
NumPy
Pillow
LPIPS (opcional para inferência)

Para instalar LPIPS no ambiente virtual:

source /home/unicornio/User/.venv/bin/activate
python -m pip install lpips

Antes de treinar, confirme:

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

6. Estrutura dos datasets

O arquivo dataload/llie_dataset.py seleciona automaticamente as pastas a partir de --dataset_name.

6.1 LSD

Estrutura esperada:

/home/unicornio/User/DataSetsLLIE/LSD/
|
|-- inputPatchDLL/
|-- gtPatchDLL/
`-- Testing/
    `-- In-the-wild/
        `-- DEI/
            |-- DEI_LOW/
            `-- DEI_GT/

Treinamento:

inputPatchDLL -> LOW
gtPatchDLL    -> GT

Validação padrão:

DEI_LOW -> LOW
DEI_GT  -> GT

6.2 LOL-v1

/home/unicornio/User/DataSetsLLIE/LOLv1/
|
|-- our485/
|   |-- low/
|   `-- high/
|
`-- eval15/
    |-- low/
    `-- high/

6.3 LOL-v2 Real

/home/unicornio/User/DataSetsLLIE/LOLv2/Real_captured/
|
|-- Train/
|   |-- Low/
|   `-- Normal/
|
`-- Test/
    |-- Low/
    `-- Normal/

6.4 LOL-v2 Synthetic

/home/unicornio/User/DataSetsLLIE/LOLv2/Synthetic/
|
|-- Train/
|   |-- Low/
|   `-- Normal/
|
`-- Test/
    |-- Low/
    `-- Normal/

7. Nomes aceitos em --dataset_name

lsd
pamazonia
lolv1
lolv2_real
lolv2_synthetic

Exemplo:

--dataset_name lolv1

O objetivo é não manter arquivos de treinamento separados por dataset.

8. Normalização

A nova arquitetura usa exclusivamente:

RGB float32 em [0,1]

Não deve existir conversão do dataset para [-1,1].

O dataloader também não utiliza ColorJitter, alterações de gamma ou augmentações cromáticas. As augmentações são apenas geométricas e sincronizadas entre LOW e GT:

flip horizontal
flip vertical
rotação 0, 90, 180 ou 270 graus
crop sincronizado

Esse comportamento é importante para não introduzir artificialmente diferenças de cor entre LOW e GT.

9. Treinamento

9.1 Treinamento recomendado para LSD

torchrun --standalone --nproc_per_node=3 \
/home/unicornio/User/LCWNet/train_lcw_hvi.py \
--dataset_name lsd \
--dataset_root /home/unicornio/User/DataSetsLLIE \
--patch_size 128 \
--patches_per_image 16 \
--batch_size 16 \
--num_workers 8 \
--channels 48 \
--num_heads 4 \
--depth 4 \
--wavelet_mode on \
--color_mode lock \
--color_unlock_epoch 31 \
--color_scale 0.03 \
--curve_steps 4 \
--curve_scale 1.0 \
--hvi_k 0.2 \
--epochs 500 \
--lr 5e-5 \
--min_lr 1e-6 \
--warmup_epochs 5 \
--grad_clip 1.0 \
--val_every 1 \
--val_max_side 1024 \
--checkpoints_dir /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LSD

9.2 Protocolo cromático recomendado

Durante as primeiras 30 épocas:

color_mode = lock

A rede pode modificar a intensidade, mas não pode modificar H e V.

A partir da época 31:

color_mode = bounded

Isso é controlado automaticamente por:

--color_mode lock
--color_unlock_epoch 31

Antes de aceitar a liberação de cor, avalie visualmente e quantitativamente epoch_0030.pth.

Caso ainda exista alteração cromática forte com color_mode=lock, não avance para bounded. Nesse cenário, investigue primeiramente o pipeline RGB, pareamento LOW/GT e transformação HVI.

Para manter a cor bloqueada durante todas as épocas:

--color_mode lock
--color_unlock_epoch -1

Para permitir correção cromática desde a primeira época:

--color_mode bounded

Essa última configuração não é recomendada para o primeiro experimento.

9.3 LOL-v1

O mesmo arquivo é usado. Altere apenas o dataset e o diretório de checkpoints:

torchrun --standalone --nproc_per_node=3 \
/home/unicornio/User/LCWNet/train_lcw_hvi.py \
--dataset_name lolv1 \
--dataset_root /home/unicornio/User/DataSetsLLIE \
--patch_size 128 \
--patches_per_image 16 \
--batch_size 16 \
--num_workers 8 \
--channels 48 \
--num_heads 4 \
--depth 4 \
--wavelet_mode on \
--color_mode lock \
--color_unlock_epoch 31 \
--epochs 500 \
--checkpoints_dir /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LOLv1

9.4 LOL-v2 Real

--dataset_name lolv2_real

9.5 LOL-v2 Synthetic

--dataset_name lolv2_synthetic

10. Checkpoints

O treinamento produz:

latest.pth
best_psnr.pth
best_ssim.pth
best_chroma.pth
epoch_XXXX.pth
train_log.csv

best_chroma.pth corresponde ao menor erro de magnitude cromática observado durante a validação.

Para avaliação científica final, compare pelo menos:

best_psnr.pth
best_ssim.pth
best_chroma.pth

Não avalie apenas latest.pth.

11. Retomada de treinamento

Exemplo:

torchrun --standalone --nproc_per_node=3 \
/home/unicornio/User/LCWNet/train_lcw_hvi.py \
--dataset_name lsd \
--dataset_root /home/unicornio/User/DataSetsLLIE \
--epochs 500 \
--batch_size 16 \
--patch_size 128 \
--patches_per_image 16 \
--channels 48 \
--num_heads 4 \
--depth 4 \
--wavelet_mode on \
--color_mode lock \
--color_unlock_epoch 31 \
--checkpoints_dir /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LSD \
--resume /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LSD/latest.pth

Parâmetros estruturais precisam coincidir com o checkpoint.

Não use checkpoints da LCWSwinNet antiga na LCWHVINet.

12. Métricas monitoradas no treinamento

O CSV registra:

loss_total
loss_rgb
loss_intensity
loss_hv
loss_chroma
loss_hue
loss_grad
loss_curve_smooth
loss_color_delta
loss_ssim
grad_norm
curve_abs_mean
delta_hv_abs_mean
train_high_clip_fraction
val_psnr
val_ssim
val_intensity_mae
val_chroma_mae
val_high_clip_fraction

Para o problema de cor estourada, monitore principalmente:

val_chroma_mae
delta_hv_abs_mean
val_high_clip_fraction

Durante color_mode=lock:

delta_hv_abs_mean = 0

é o comportamento esperado.

13. Inferência

O arquivo infer_lcw_hvi.py também é único para todos os datasets.

13.1 Inferência automática no LSD/DEI

Se --input_path não for informado, o script localiza automaticamente o conjunto de validação:

python /home/unicornio/User/LCWNet/infer_lcw_hvi.py \
--checkpoint /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LSD/best_psnr.pth \
--dataset_name lsd \
--dataset_root /home/unicornio/User/DataSetsLLIE \
--output_dir /home/unicornio/User/LCWNet/results/LCWHVINet_LSD_DEI \
--device cuda \
--tile_size 0

13.2 Inferência LOL-v1

python /home/unicornio/User/LCWNet/infer_lcw_hvi.py \
--checkpoint /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LOLv1/best_psnr.pth \
--dataset_name lolv1 \
--dataset_root /home/unicornio/User/DataSetsLLIE \
--output_dir /home/unicornio/User/LCWNet/results/LCWHVINet_LOLv1 \
--device cuda \
--tile_size 0

13.3 Inferência em uma pasta arbitrária

python /home/unicornio/User/LCWNet/infer_lcw_hvi.py \
--checkpoint /home/unicornio/User/LCWNet/ckpt/LCWHVINet_LSD/best_psnr.pth \
--dataset_name lsd \
--input_path /caminho/LOW \
--gt_path /caminho/GT \
--output_dir /caminho/resultados \
--device cuda

Quando --gt_path é omitido, a rede gera as imagens, mas PSNR, SSIM, LPIPS, Intensity-MAE e Chroma-MAE não são calculados.

14. Inferência por tiles

Para uma imagem que cabe integralmente na GPU, use:

--tile_size 0

Essa configuração deve ser utilizada nos primeiros testes de artefatos, porque elimina a composição por tiles como variável experimental.

Para imagens 4K que ultrapassem a memória disponível:

--tile_size 512
--tile_overlap 64

O script usa uma janela de Hann para misturar regiões sobrepostas.

Evite tiles muito pequenos durante a avaliação qualitativa. Um tile de 128 pixels pode restringir excessivamente o contexto espacial e introduzir descontinuidades visuais.

15. AMP

AMP está desligado por padrão.

Para ativar:

--amp

Nos primeiros experimentos relacionados a estabilidade de cor, recomenda-se executar sem AMP. Isso reduz uma variável numérica adicional durante o diagnóstico.

16. LPIPS

LPIPS é calculado quando existe GT.

Para desativar:

--disable_lpips

Para escolher o backbone:

--lpips_net alex

ou:

--lpips_net vgg

Para imagens grandes, a avaliação LPIPS pode ser limitada:

--lpips_max_size 256

17. Saídas da inferência

O diretório de resultados contém:

imagem_LCWHVI.png
metrics.csv
metrics_summary.txt

O CSV registra:

PSNR
SSIM
LPIPS
Intensity MAE
Chroma MAE
fração de pixels >= 0.999
tempo de inferência
média da saída
caminho da imagem salva

A fração de pixels maiores ou iguais a 0.999 deve ser monitorada para detectar clipping excessivo de altas luzes.

18. Execução no servidor com Singularity

O projeto utilizado neste ambiente está em:

/home/unicornio/User/LCWNet

Imagem Singularity:

/home/unicornio/Python.sif

Ambiente virtual:

/home/unicornio/User/.venv

A execução pode ser encapsulada em um arquivo .sh dentro de:

/home/unicornio/User/gpuFarm

19. Arquivo .sh de treinamento

Exemplo fornecido:

train_lcw_hvi.sh

Torne-o executável:

chmod +x /home/unicornio/User/gpuFarm/train_lcw_hvi.sh

Teste dentro do Singularity:

singularity exec --nv \
-B /home/unicornio \
-B /scratch \
--pwd /home/unicornio/User/LCWNet \
/home/unicornio/Python.sif \
bash /home/unicornio/User/gpuFarm/train_lcw_hvi.sh

20. Execução direta com srun

Exemplo em três GPUs de um único nó:

srun -N1 -n1 \
--cpu_bind=cores \
--nodelist=gn02 \
--cpus-per-task=64 \
singularity exec --nv \
-B /home/unicornio \
-B /scratch \
--pwd /home/unicornio/User/LCWNet \
/home/unicornio/Python.sif \
bash /home/unicornio/User/gpuFarm/train_lcw_hvi.sh

O CUDA_VISIBLE_DEVICES=0,1,2 e o torchrun --nproc_per_node=3 já podem permanecer dentro do .sh.

21. Arquivo .slurm

Um arquivo .slurm é utilizado com sbatch.

Exemplo:

sbatch /home/unicornio/User/gpuFarm/train_lcw_hvi.slurm

Consultar jobs:

squeue -u unicornio

Cancelar:

scancel ID_DO_JOB

A diretiva:

#SBATCH --gres=gpu:a100:3

solicita três A100 quando essa sintaxe estiver configurada no cluster. Caso a GPU-FARM utilize outro nome de recurso, altere apenas essa diretiva.

22. .sh versus .ssh

.sh é um script Shell executável. É o formato utilizado para guardar os comandos de treinamento e inferência.

.ssh normalmente não é uma extensão de script. O diretório:

~/.ssh/

armazena chaves e configuração do SSH.

Portanto, para executar a LCWHVINet no servidor, normalmente são necessários:

train_lcw_hvi.sh
train_lcw_hvi.slurm

e não um arquivo .ssh.

23. Configuração SSH opcional

Caso deseje simplificar o acesso ao servidor, crie:

~/.ssh/config

com estrutura semelhante a:

Host pavicgpufarm
    HostName ENDERECO_REAL_DO_SERVIDOR
    User unicornio
    IdentityFile ~/.ssh/SUA_CHAVE_PRIVADA
    IdentitiesOnly yes

Depois ajuste permissões:

chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/SUA_CHAVE_PRIVADA

A chave pública pode permanecer com:

chmod 644 ~/.ssh/SUA_CHAVE_PRIVADA.pub

Nunca envie a chave privada para GitHub, repositórios ou outras pessoas.

Depois:

ssh pavicgpufarm

Não preencha HostName ou IdentityFile com valores inventados. Utilize o host real fornecido pela administração da GPU-FARM e a chave privada efetivamente cadastrada para acesso ao servidor.

24. Organização recomendada no servidor

/home/unicornio/User/
|
|-- LCWNet/
|   |-- models/
|   |-- dataload/
|   |-- train_lcw_hvi.py
|   |-- infer_lcw_hvi.py
|   |-- ckpt/
|   `-- results/
|
|-- DataSetsLLIE/
|   |-- LSD/
|   |-- LOLv1/
|   `-- LOLv2/
|
|-- gpuFarm/
|   |-- train_lcw_hvi.sh
|   |-- infer_lcw_hvi.sh
|   |-- train_lcw_hvi.slurm
|   `-- infer_lcw_hvi.slurm
|
`-- .venv/

25. Verificações antes do primeiro treinamento

Execute:

cd /home/unicornio/User/LCWNet
source /home/unicornio/User/.venv/bin/activate
python -m py_compile models/lcw_hvi_backbone.py
python -m py_compile models/loss_hvi.py
python -m py_compile dataload/llie_dataset.py
python -m py_compile train_lcw_hvi.py
python -m py_compile infer_lcw_hvi.py

Depois confirme importações:

python -c "from models.lcw_hvi_backbone import LCWHVINet; print('Backbone OK')"
python -c "from models.loss_hvi import LCWHVITotalLoss; print('Loss OK')"
python -c "from dataload.llie_dataset import LLIETrainDataset; print('Dataset OK')"

26. Protocolo inicial recomendado

Para o primeiro experimento científico:

mantenha wavelet_mode=on;

mantenha color_mode=lock;

configure color_unlock_epoch=31;

treine com patches 128x128;

acompanhe epoch_0005, epoch_0010, epoch_0020 e epoch_0030;

execute inferência inicialmente com tile_size=0 em uma imagem que caiba na GPU;

compare PSNR, SSIM, LPIPS, Intensity-MAE, Chroma-MAE e clipping;

confirme visualmente que a cor permanece estável;

somente então permita a fase bounded;

compare best_psnr, best_ssim e best_chroma.

Esse protocolo separa experimentalmente o problema de iluminação do problema cromático e reduz a quantidade de alterações simultâneas durante o diagnóstico.

27. Observação de compatibilidade

A arquitetura possui a identificação:

LCWHVINet_HVI_Restormer_v1

Os scripts recusam checkpoints de versões anteriores. Isso é intencional.

Não use checkpoints da LCWSwinNet baseada em Swin, base_head, residual_head, downscaling ou faixa [-1,1].

28. Arquivos de log

Para acompanhar um job submetido pelo SLURM:

tail -f /home/unicornio/User/gpuFarm/lcwhvi_train_JOBID.out

ou, conforme o arquivo .slurm fornecido:

tail -f /home/unicornio/User/gpuFarm/lcwhvi_train_JOBID.err

O CSV do treinamento permanece em:

CHECKPOINTS_DIR/train_log.csv

e deve ser preservado junto com os checkpoints utilizados no artigo e nos estudos de ablação.