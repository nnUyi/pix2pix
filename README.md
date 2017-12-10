# pix2pix
  - A implement of pix2pix for tensorflow version according to paper named [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)

# Requirements
  - tensorflow 1.3.0

  - python 2.7.12

  - numpy 1.13.1

  - scipy 0.17.0
  
# Usages
  ## download repo
      $ git clone https://github.com/nnuyi/pix2pix
      $ cd pix2pix
      
  ## download datasets
  - This is repo mainly focus on **facades** datasets, more datasets you can download [here](https://phillipi.github.io/pix2pix/), or you just need to run dataset_download.sh as follow:
  
      #dataset_download.sh is provided to download datasets
      $ bash ./dataset_download ***
      # here *** indicates dataset name, like facades or Cityscapes etc, example show below
      $ bash ./dataset_download.sh facades
      
  - Then, you can see the dataset named facades is stored in your datasets(named datasets) directory  
  ## training model
  ## testing model
  
# Contacts
  - Email:computerscienceyyz@163.com
