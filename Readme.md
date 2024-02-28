# Federated Learning Simulator
+ Method: FedAvg
+ Pending: FedISM (Shared Model + Balanced CSM)
+ Future: FedProx, FedNova, Scaffold

# Clone repo
```bash
git clone https://github.com/huihui0310tt/Federated-Learning.git
```
``` bash
# cd origindata
nano configure.json
nano command.sh
nano net.py (num_classes)

chmod +x createnew.sh
chmod +x command.sh

./createnew.sh
./command.sh
```


# Environment
1. Install Anaconda
   ```bash
    cd /tmp 

    curl https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh --output anaconda.sh

    bash anaconda.sh
        ..
        ..
        ..
    source ~/.bashrc

   ```
2. Create Environment
   ```bash
     conda create --name hui_icce2024_python37 python=3.7
    # Activate
    conda activate hui_icce2024_python37
    # REMOVE
    conda env remove --name hui_icce2024_python37
   ```
3. Install PyTorch
   ```bash
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
   ```

4. Install Package
    ```
    sudo apt install -y jq
    pip install tabulate
    ```
