# pix2pix
  - An implement of pix2pix for tensorflow version according to paper named [Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)

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
      # here facades indicates dataset name, you can change to Cityscapes etc, example shows below
      $ bash ./dataset_download.sh facades    
  
  - This is repo mainly focus on **facades** datasets, more datasets you can download [here](https://phillipi.github.io/pix2pix/), or you just need to run dataset_download.sh,then, you can see the dataset named facades is stored in your datasets(named datasets) directory.
  
  ## training model
      $ python main.py --is_training True --phase train
      
  ## testing model
      $ python main.py --is_testing True --phase test

# Results
  ## sample data from val sets(sampling)
  |sample result|sample result|
  |:-----------------:|:----------------:|
  |![Alt test](/data/facades_train_1.png)|![Alt test](/data/facades_train_2.png)|
  |left:source<br/>middle: groundtruth<br/>right:sample|left:source<br/>middle: groundtruth<br/>right:sample||
  
  ## sample data from test sets(testing)
  |test result|test result|
  |:-----------------:|:----------------:|
  |![Alt test](/data/facades_test_1.png)|![Alt test](/data/facades_test_2.png)|
  |left:source<br/>middle: groundtruth<br/>right:sample|left:source<br/>middle: groundtruth<br/>right:sample||
  |![Alt test](/data/facades_test_3.png)|![Alt test](/data/facades_test_4.png)|
  |left:source<br/>middle: groundtruth<br/>right:sample|left:source<br/>middle: groundtruth<br/>right:sample||

# Acknowledgements
  - dataset_down.sh is obtained from [phillipi/pix2pix](https://github.com/phillipi/pix2pix/tree/master/datasets)

# Contacts
  - Email:computerscienceyyz@163.com
