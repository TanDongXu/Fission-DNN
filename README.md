>Fission-DNN
>==========
>Results
>--------
>CNN accelerated by cuda and lib <a href ="https://developer.nvidia.com/cudnn">CUDNN v5</a>

> 1. Test on <a href="http://yann.lecun.com/exdb/mnist/"> mnist</a>    
> 2. Test on cifar-10

>Feature
>--------
>1. Use cudnn lib to develop CNN
>2. Learn from the <a href="https://github.com/BVLC/caffe"> Caffe</a> structure of Blob,modedify the structure of the first <a href="https://github.com/TanDongXu/CUDA-MCDNN">First version.</a>

>Compile
>-------
>1. Depend on opencv, google protobuf, google glog, LMDB, cudnn and cuda    
>2. You can compile the code on windows or linux.   
>3. Some library install:

>sudo  apt-get install libprotobuf-dev libopencv-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev liblmdb-dev

###GPU compute 
>* capability 2.0   

###CMake for Linux
>1. mkdir build  
>2. cd build  
>3. cmake ..  
>4. make -j4  
>5. cd ../mnist/  
>6. sh get_mnist.sh  
>7. cd ../cifar-10  
>8. sh get_cifar10.sh  
>9. cd ../  
>10. ./build/Fission-DNN  

>Informations
>------------
>* Author : TDX  
>* Mail   :sa614149@mail.ustc.edu.cn  
>* Welcome for any suggest!!   

