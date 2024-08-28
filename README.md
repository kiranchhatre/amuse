<p align="center">
  <img width="33%" src="docs/static/amuse_icon.png">
</p>

---

<p align="center">
<a href="https://www.kth.se/profile/chhatre"><strong>Kiran Chhatre</strong></a>
·
    <a href="https://ps.is.tuebingen.mpg.de/person/rdanecek"><strong>Radek Daněček</strong></a>    
    ·
    <a href="https://ps.is.mpg.de/person/nathanasiou"><strong>Nikos Athanasiou</strong></a>
    <br>
    <a href="https://ps.is.mpg.de/person/gbecherini"><strong>Giorgio Becherini</strong></a>
    ·
    <a href="https://www.kth.se/profile/chpeters"><strong>Christopher Peters</strong></a>
    ·
    <a href="https://ps.is.tuebingen.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    ·
    <a href="https://sites.google.com/site/bolkartt"><strong>Timo Bolkart</strong></a>
  </p>

<p align="center">
  <br>
  <a href='https://amuse.is.tue.mpg.de/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/AMUSE-Page-pink?style=for-the-badge&logo=Google%20chrome&logoColor=pink' alt='Project Page'>
    </a>
     <a href='https://arxiv.org/abs/2312.04466'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <a href='https://youtu.be/gsEt9qtR1jk' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Intro-Video-orange?style=for-the-badge&logo=youtube&logoColor=orange' alt='Intro Video'>
    </a>
    <a href='https://drive.google.com/file/d/1-FRWKlW9fr5y-lrija4E12gnm_A13MzN/view?usp=sharing' style='padding-left: 0.5rem;'>
  <img src='https://img.shields.io/badge/Poster-PDF-blue?style=for-the-badge&logo=Adobe%20Acrobat%20Reader&logoColor=blue' alt='Poster PDF'>
</a>
  </p>
</p>
<br/>

<p align="center">
  <img width="50%" src="docs/static/drawing_v4.svg">
</p>



<br/>

This is a repository for **AMUSE**: Emotional Speech-driven 3D Body Animation via Disentangled Latent Diffusion. AMUSE generates realistic emotional 3D body gestures directly from a speech sequence (*top*). It provides user control over the generated emotion by combining the driving speech with a different emotional audio (*bottom*).
<br/> 

## News :triangular_flag_on_post:

- [2024/07/25] Data processing and gesture editing scripts are available. 
- [2024/06/12] Code is available.
- [2024/02/27] AMUSE has been accepted for CVPR 2024! Working on code release.
- [2023/12/08] <a href="https://arxiv.org/abs/2312.04466">ArXiv</a> is available.

---

## Setup

### Main Repo Setup

The project has been tested with the following configuration:

- **Operating System**: Linux 5.14.0-1051-oem x86_64
- **GCC Version**: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
- **CUDA Version**: CUDA 11.3
- **Python Version**: Python 3.8.15
- **GPU Configuration**:
  - **Audio Model**: NVIDIA A100-SXM4-80GB
  - **Motion Model**: NVIDIA A100-SXM4-40GB, Tesla V100-32GB

**Note**: The audio model requires a larger GPU. Multiple GPU support is implemented for the audio model; however, it was not used in the final version.


```bash
git clone https://github.com/kiranchhatre/amuse.git
cd amuse/dm/utils/
git clone https://github.com/kiranchhatre/sk2torch.git
git clone -b init https://github.com/kiranchhatre/PyMO.git
cd ../..
git submodule update --remote --merge --init --recursive
git submodule sync

git submodule add https://github.com/kiranchhatre/sk2torch.git dm/utils/sk2torch
git submodule add -b init https://github.com/kiranchhatre/PyMO.git dm/utils/PyMO

git submodule update --init --recursive

git add .gitmodules dm/utils/sk2torch dm/utils/PyMO
```

### Environment Setup

```bash
conda create -n amuse python=3.8
conda activate amuse
export CUDA_HOME=/is/software/nvidia/cuda-11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda env update --file amuse.yml --prune
module load cuda/11.3
conda install anaconda::gxx_linux-64 # install 11.2.0
FORCE_CUDA=1 pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

### Blender Setup

```bash
conda deactivate
conda env create -f blender.yaml
AMUSEPATH=$(pwd)
cd ~
wget https://download.blender.org/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz
tar -xvf ./blender-3.4.1-linux-x64.tar.xz
cd ~/blender-3.4.1-linux-x64/3.4
mv python/ _python/
ln -s /home/kchhatre/anaconda3/envs/envs/blender ./python
cd "$AMUSEPATH"
cd scripts
conda activate amuse
```


---


## Data Setup and Blender Resources

Follow instructions: [https://amuse.is.tue.mpg.de/download.php](https://amuse.is.tue.mpg.de/download.php)


---


## Tasks

Once the above setup is correctly done, you can execute the following:

- [x] **train_audio (training step 1/2)**  
  Train AMUSE step 1 of the speech disentanglement model.
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn train_audio
  ```

- [x] **train_gesture (training step 2/2)**  
  Train AMUSE step 2 of the gesture generation model.
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn train_gesture
  ```

- [x] **infer_gesture**  
  Infer AMUSE on a single 10s WAV monologue audio sequence.  
  Place audio in `$AMUSEPATH/viz_dump/test/speech`.  
  Video of generated gesture will be in `$AMUSEPATH/viz_dump/test/gesture`.
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn infer_gesture
  ```

- [x] **edit_gesture**  
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn edit_gesture
  ```
  For extensive editing options, please refer to the `process_loader` function in `infer_ldm.py` and experiment with different configurations in `emotion_control`, `style_transfer`, and `style_Xemo_transfer`. While editing gestures directly from speech is challenging, it offers intriguing possibilities. The task involves numerous combinations, and not all may yield optimal results. Figures A.11 and A.12 in supplementary material illustrate the inherent complexities and variations in this process.
  Click the image below to watch the video on YouTube:
  <div align="center">
  <a href="https://youtu.be/48vw2NfWkJg" target="_blank">
    <img src="https://img.youtube.com/vi/48vw2NfWkJg/maxresdefault.jpg" alt="Video Thumbnail">
  </a>
  </div>


- [x] **bvh2smplx_**  
  The BVH import process can cause inaccuracies due to Euler angle singularities, where different sets of rotation angles correspond to the same rotation. We hacked this issue by modifying the Blender source code within the `import_bvh.py` file of the `io_anim_bvh` addon, although it's not a perfect solution. Specifically, in the `dm.beat2smplnpz()` function, we made the following change in `<YOUR-PATH>/blender-3.6.0-linux-x64/3.6/scripts/addons/io_anim_bvh/import_bvh.py`:

  - **Lines 636-637:** Toggled the rotation argument from `True` to `False`:
    - **Before:** `bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)`
    - **After:** `bpy.ops.object.transform_apply(location=False, rotation=False, scale=False)`

  Convert BVH to SMPL-X using the provided BMAP presets from the AMUSE website download page. Place the BVH file inside `$AMUSEPATH/data/beat-rawdata-eng/beat_rawdata_english/<<actor_id>>`, where `actor_id` is a number between 1 and 30. The converted file will be located in `$AMUSEPATH/viz_dump/smplx_conversions`.
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn bvh2smplx_
  ```
  Once converted, import the file in Blender using the SMPLX blender addon. Remember to specify the target FPS (for current file: 24 FPS) in the import animation window while importing the NPZ file.

<p align="center">
  <img width="50%" src="docs/static/BVH2SMPLX.gif">
</p>

- [x] **prepare_data**  
  Prepare data and create an LMDB file for training AMUSE. We provide the AMUSE-BEAT version on the project webpage. To train AMUSE on a custom dataset, you will need aligned motion and speech files. The motion data should be in an animation NPZ file compatible with the SMPL-X format.
  ```bash
  cd $AMUSEPATH/scripts
  python main.py --fn prepare_data
  ```

---

## Citation

If you find the Model & Software, BVH2SMPLX conversion tool, and SMPLX Blender addon-based visualization software useful in your research, we kindly ask that you cite our work:

```bibtex
@InProceedings{Chhatre_2024_CVPR,
    author    = {Chhatre, Kiran and Daněček, Radek and Athanasiou, Nikos and Becherini, Giorgio and Peters, Christopher and Black, Michael J. and Bolkart, Timo},
    title     = {{AMUSE}: Emotional Speech-driven {3D} Body Animation via Disentangled Latent Diffusion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {1942-1953},
    url = {https://amuse.is.tue.mpg.de},
}
```

Additionally, if you use the [AMUSE-BEAT data](https://amuse.is.tue.mpg.de/download.php) in your research, please also consider citing both the AMUSE and [EMAGE](https://pantomatrix.github.io/EMAGE/) projects.

<br/>

## License

Software Copyright License for non-commercial scientific research purposes.
Please read carefully the following [terms and conditions](LICENSE) and any accompanying
documentation before you download and/or use AMUSE model, AMUSE-BEAT data and
software, (the "Data & Software"), including 3D meshes, images, videos,
textures, software, scripts, and animations. By downloading and/or using the
Data & Software (including downloading, cloning, installing, and any other use
of the corresponding github repository), you acknowledge that you have read
these [terms and conditions](LICENSE), understand them, and agree to be bound by them. If
you do not agree with these [terms and conditions](LICENSE), you must not download and/or
use the Data & Software. Any infringement of the terms of this agreement will
automatically terminate your rights under this [License](LICENSE).

<br/>

## Acknowledgments

We would like to extend our gratitude to the authors and contributors of the following open-source projects, whose work has significantly influenced and supported our implementation: [EVP](https://github.com/jixinya/EVP), [Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model), [Motion Latent Diffusion](https://github.com/ChenFengYe/motion-latent-diffusion), [AST](https://github.com/YuanGongND/ast), [ACTOR](https://github.com/Mathux/ACTOR), and [SMPL-X](https://github.com/vchoutas/smplx). We also wish to thank [SlimeVRX](https://github.com/SlimeVRX) for their collaboration on the development of the `bvh2smplx_` task. For a more detailed list of acknowledgments, please refer to our paper. 

<br/>

## Contact

For any inquiries, please feel free to contact [amuse@tue.mpg.de](mailto:amuse@tue.mpg.de). Feel free to use this project and contribute to its improvement. For commercial uses of the Data & Software, please send an email to [ps-license@tue.mpg.de](mailto:ps-license@tue.mpg.de).