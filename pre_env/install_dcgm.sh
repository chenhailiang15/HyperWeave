git clone https://github.com/NVIDIA/DCGM.git
cd DCGM
./build.sh
sudo make install
sudo systemctl start dcgm