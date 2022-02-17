# TRACER: Extreme Attention Guided Salient Object Tracing Network

This paper was accepted at AAAI 2022 SA poster session. [[pdf]](https://arxiv.org/abs/2112.07380)    

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-duts-te)](https://paperswithcode.com/sota/salient-object-detection-on-duts-te?p=tracer-extreme-attention-guided-salient)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-dut-omron)](https://paperswithcode.com/sota/salient-object-detection-on-dut-omron?p=tracer-extreme-attention-guided-salient)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-hku-is)](https://paperswithcode.com/sota/salient-object-detection-on-hku-is?p=tracer-extreme-attention-guided-salient)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-ecssd)](https://paperswithcode.com/sota/salient-object-detection-on-ecssd?p=tracer-extreme-attention-guided-salient)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tracer-extreme-attention-guided-salient/salient-object-detection-on-pascal-s)](https://paperswithcode.com/sota/salient-object-detection-on-pascal-s?p=tracer-extreme-attention-guided-salient) 

![alt text](https://github.com/Karel911/TRACER/blob/main/img/Poster.png)


## Datasets
All datasets are available in public.
* Download the DUTS-TR and DUTS-TE from [Here](http://saliencydetection.net/duts/#org3aad434)
* Download the DUT-OMRON from [Here](http://saliencydetection.net/dut-omron/#org96c3bab)
* Download the HKU-IS from [Here](https://sites.google.com/site/ligb86/hkuis)
* Download the ECSSD from [Here](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
* Download the PASCAL-S from [Here](http://cbs.ic.gatech.edu/salobj/)
* Download the edge GT from [Here](https://drive.google.com/file/d/1Xl-OwmbkmB1dnvIrcLq3OPQjQnieynpk/view?usp=sharing).

## Data structure
<pre><code>
TRACER
├── data
│   ├── DUTS
│   │   ├── Train
│   │   │   ├── images
│   │   │   ├── masks
│   │   │   ├── edges
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── DUT-O
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
│   ├── HKU-IS
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── masks
      .
      .
      .
</code></pre>

## Requirements
* Python >= 3.7.x
* Pytorch >= 1.8.0
* albumentations >= 0.5.1
* tqdm >=4.54.0
* scikit-learn >= 0.23.2

## Run
* Run **main.py** scripts.
<pre><code>
# For training TRACER-TE0 (e.g.)
python main.py train --arch 0 --img_size 320

# For testing TRACER with pre-trained model (e.g.)  
python main.py test --exp_num 0 --arch 0 --img_size 320
</code></pre>
* Pre-trained models of TRACER are available at [here](https://github.com/Karel911/TRACER/releases/tag/v1.0)
* Change the model name as 'best_model.pth' and put the weights to the path 'results/DUTS/TEx_0/best_model.pth'  
  (here, the x means the model scale e.g., 0 to 7).
* Input image sizes for each model are listed belows.

## Configurations
--arch: EfficientNet backbone scale: TE0 to TE7.  
--frequency_radius: High-pass filter radius in the MEAM.  
--gamma: channel confidence ratio \gamma in the UAM.   
--denoise: Denoising ratio d in the OAM.  
--RFB_aggregated_channel: # of channels in receptive field blocks.  
--multi_gpu: Multi-GPU learning options.  
--img_size: Input image resolution.  
--save_map: Options saving predicted mask.  

<table>
<thead>
  <tr>
    <th>Model</th>
    <th>Img size</th>
  </tr>
</thead>
<tbody>
    <tr>
        <td>TRACER-Efficient-0 ~ 1</td>
        <td>320</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-2</td>
        <td>352</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-3</td>
        <td>384</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-4</td>
        <td>448</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-5</td>
        <td>512</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-6</td>
        <td>576</td>
    </tr>
    <tr>
        <td>TRACER-Efficient-7</td>
        <td>640</td>
    </tr>
</tbody>
</table>

## Citation
<pre><code>
@article{lee2021tracer,
  title={TRACER: Extreme Attention Guided Salient Object Tracing Network},
  author={Lee, Min Seok and Shin, WooSeok and Han, Sung Won},
  journal={arXiv preprint arXiv:2112.07380},
  year={2021}
}
</code></pre>
