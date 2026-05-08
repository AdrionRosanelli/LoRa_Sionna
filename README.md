# LoRa_Sionna

### Implementação da camada física do LoRa baseado no Sionna e TensorFlow.

* Arquivo "Lora_Phy_Tx_Batch.ipynb": Arquivo com a simulação do Transmissor e analises.
  - Executado em um ambiente WSL.

### Análise da implementação da camada física do LoRa no GNU Radio.

* Arquivos "gr_cell1_time_benchmark.py", "gr_cell2_resource_benchmark.py", "gr_cell3_step_profiling.py": rotinas para analises da simulação do GNU.
  - Executado em um ambiente do radioconda (GNU Radio) pelo seguinte comando:
    ```& C:/Users/aaros/radioconda/python.exe "c:/Adrion/Doutorado/Sionna - LoRa/Simulador LoRa Phy/gr-lora_sdr-master/gr_cell1_time_benchmark.py"```
  - Esses arquivos devem estar na pasta do repositório do Tapparel clonado ( https://github.com/tapparelj/gr-lora_sdr.git )

### Comparação das simulações.

* Arquivo "plot_comparison.py": Arquivo para geração de gráficos de comparação.
  - Executado em qualquer ambiente python, mas precisa dos arquivos .csv gerados nas análises descritas acima.

### Para testes.
* Arquivo "Sionna_tutorial_part1.ipynb" tutorial inicial do Sionna PHY.
