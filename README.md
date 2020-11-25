## Albumentations使用

仓库：https://github.com/albumentations-team/albumentations

手册：https://albumentations.ai/docs/

### 1.输入数据

图片：

```python
# opencv读入或者PIL
image = cv2.imread('000009.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

物体框：

```python
# yolo格式 也可以VOC COCO
bboxes = [
    [0.379, 0.5666666666666667, 0.158, 0.3813333333333333],
    [0.612, 0.7093333333333333, 0.084, 0.3466666666666667],
    [0.555, 0.7026666666666667, 0.078, 0.34933333333333333]
]
```

类别：

```python
# 类别和bboxes框写在一起，此时不需要额外的类别标签
# 可以是整数  字符串 
bboxes = [
    [0.379, 0.5666666666666667, 0.158, 0.3813333333333333, 'person'],
    [0.612, 0.7093333333333333, 0.084, 0.3466666666666667, 'person'],
    [0.555, 0.7026666666666667, 0.078, 0.34933333333333333, 'person']
]

# 可以直接传名称
# 可以是整数  字符串
category_ids = ['person', 'person', 'person']
```

### 2.定义变换

```python
import albumentations as A

# A.Compose()传入变换的列表 和 检测框的参数
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=1),    
        A.RandomGamma(p=1),    
        A.CLAHE(p=1),   
    ], 
    bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids'])
)
```

### 3.变换

```python
# 传入要做变换的图片 及其对应的 物体框和类别
transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)

# 得到一个字典 保存变换后的图片 及其对应的 物体框和类别
transformed['image']
transformed['bboxes']
transformed['category_ids']
```



### 补充：常用变换

#### 1.方向变换

- **Flip**(p=0.5) 水平，垂直 或 水平和垂直 翻转输入 

- **HorizontalFlip**(p=0.5)  水平翻转 

- **RandomRotate90**(p=1) 将输入随机旋转90度，零次或多次。

- **Rotate** (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)

- **Transpose**() 置换行列即顺时针转90°

- **VerticalFlip**() 垂直翻转


#### 2.大小变换

- **LongestMaxSize** (max_size=1024, interpolation=1, always_apply=False, p=1)重新缩放图像，以使最大边等于max_size，并保持初始图像的纵横比。

- **SmallestMaxSize** (max_size=1024, interpolation=1, always_apply=False, p=1) 重新缩放图像，以使最小边等于max_size，并保持初始图像的纵横比。

- **RandomScale** (scale_limit=0.1, interpolation=1, always_apply=False, p=0.5) 随机调整输入大小。输出图像尺寸与输入图像尺寸不同。 范围将是（1-scale_limit，1 + scale_limit）

- **Resize** (height, width, interpolation=1, always_apply=False, p=1)

#### 2-1.裁剪变换

- **Crop** (x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0) 从图像中裁剪区域

- **CenterCrop** (height, width, always_apply=False, p=1.0)  裁剪输入的中心部分

- **RandomCrop** (height, width, always_apply=False, p=1.0) 随机裁剪

- **RandomResizedCrop** (w,h, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=1.0) 裁剪面积scale长宽比ratio的区域缩放到w*h

- **RandomSizedBBoxSafeCrop** (height, width, erosion_rate=0.0, interpolation=1, always_apply=False, p=1.0) 裁剪输入的随机部分，并将其重新缩放到一定大小，而不会丢失bbox。

- **RandomSizedCrop** (min_max_height, height, width, w2h_ratio=1.0, interpolation=1, always_apply=False, p=1.0)裁剪输入的随机部分min_max_height，然后将其重新缩放为一定大小。

#### 3.色调变换

- **Equalize** (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5) 均衡图像直方图 

- **CLAHE** (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5) 将 对比度受限的自适应直方图均衡 应用于输入图像。提升对比度 

- **ChannelDropout** (channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5) 在输入图像中随机删除通道。 从中选择下降通道数的范围。

- **ChannelShuffle**()随机重新排列输入RGB图像的通道。

- **ColorJitter** (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5)随机更改图像的亮度，对比度和饱和度和色调

- **HueSaturationValue** (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)随机更改输入图像的色相，饱和度和值。处于指定范围的色调和饱和度更改到val_shift的范围

- **RandomBrightnessContrast** (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5)随机更改输入图像的亮度和对比度。

- **HistogramMatching** (reference_images, blend_ratio=(0.5, 1.0), read_fn=<function read_rgb_image at 0x7f0a85c11f70>, always_apply=False, p=0.5)应用直方图匹配。它操纵输入图像的像素，以使其直方图与参考图像的直方图匹配。如果图像具有多个通道，则只要输入图像和参考图像中的通道数相等，就可以针对每个通道独立进行匹配。直方图匹配可以用作诸如特征匹配之类的图像处理的轻量级归一化，尤其是在图像是从不同来源或在不同条件下（即照明）获取的情况下。

- **RGBShift** (r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1)为输入RGB图像的每个通道随机移动值。

- **ToGray**() 灰度图

- **ToSepia**() 棕褐色滤镜

#### 4.天气变换

- **RandomFog** (fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=0.5)模拟图像雾 alpha_coef雾圈的透明度

- **RandomRain** (slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.8, rain_type=None,always_apply=False, p=0.5)添加雨水效果

- **RandomShadow** (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5)模拟图像阴影 阴影区域 数量  形状

- **RandomSnow** (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=1) 漂白一些模拟雪的像素值 降雪量 积雪量

- **RandomSunFlare** (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=200, src_color=(255, 255, 255),always_apply=False, p=1)太阳耀斑

#### 5.仿射变换

- **IAAAffine** (scale=1.0, translate_percent=None, translate_px=None, rotate=0.0, shear=0.0, order=1, cval=0, mode='reflect', always_apply=False, p=0.5) 缩放 平移百分比(像素) 旋转裁剪 cval和mode是填充方式

- **IAAPerspective** (scale=(0.05,0.2), keep_size=True, always_apply=False, p=1) 对输入执行随机四点透视变换 scale 正态分布的标准偏差。这些用于采样子图像角与整个图像角之间的随机距离。默值：（0.05，0.1）

- **IAAPiecewiseAffine** (scale=(0.03, 0.05), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=1) 扭曲：在输入上放置规则的点网格，并通过仿射变换在这些点附近随机移动 scale 确定每个点移动多远的系数范围。默认值：（0.03，0.05） nb_rows 常规网格应具有的点的行数。默认值：4。 

- **ShiftScaleRotate** (shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None,always_apply=False, p=0.5)高度和宽度的移位因子范围  比例因子范围  旋转范围 

#### 6.噪声变换

- **IAAAdditiveGaussianNoise** (loc=0, scale=(0.01*255, 0.05*255), per_channel=False, always_apply=False, p=1) 产生噪声的正态分布的平均值。默认值：0 产生噪声的正态分布的标准偏差。默认值：（0.01 * 255，0.05 * 255）

- **GaussNoise** (var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5) 将高斯噪声应用于输入图像。 

- **ISONoise** (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=False, p=1) 相机传感器的噪点。 color_shift色相变化的方差范围。在HLS色彩空间中以360度色相角的分数测量。 intensity控制颜色和亮度噪声强度的乘法因子。

- **MultiplicativeNoise** (multiplier=(0.9, 1.1), per_channel=True, elementwise=True, always_apply=False, p=1) 将图像乘以随机数或数字数组。

#### 7.边缘变换


- **IAAEmboss** (alpha=(0.2, 0.5), strength=(0.2, 0.7), always_apply=False, p=0.5) 压印输入图像，并将结果与​​原始图像重叠 alpha:选择浮雕图像可见性的范围。在0处，仅原始图像可见，在1.0处，仅其浮雕版本可见。默认值：（0.2，0.5）strength 压花强度范围。默认值：（0.2，0.7）

- **IAASharpen** (alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.5)锐化输入图像，并将结果与​​原始图像重叠。

- **Blur** (blur_limit=7, always_apply=False, p=0.5) 使用随机大小的内核(3以上 默认7)模糊输入图像

- **GaussianBlur** (blur_limit=(3,5), sigma_limit=0, always_apply=False, p=1) 使用具有随机内核大小的高斯滤波器来模糊输入图像。

- **GlassBlur** (sigma=0.7, max_delta=4, iterations=2, always_apply=False, mode='fast', p=0.5) 玻璃模糊。高斯核的标准偏差0.7。 交换像素之间的最大距离4 

- **MedianBlur** (blur_limit=7, always_apply=False, p=0.5)使用具有随机光圈线性大小的中值滤波器来模糊输入图像。 用于模糊输入图像的最大光圈线性大小。必须为奇数且在[3，inf）范围内。默认值：（3，7）

- **MotionBlur**(blur_limit=7, always_apply=False, p=0.5)使用随机大小的内核将运动模糊应用于输入图像。

#### 8.像素变换

- **IAASuperpixels** (p_replace=0.1, n_segments=100, always_apply=False, p=1)将输入图像完全或部分转换为其超像素表示。使用SLIC算法的skimage版本定义任何超像素区域被超像素（即其区域内的平均像素颜色）替换的概率。默认值：0.1目标要生成的超像素数。默认值：100。

- **Downscale** (scale_min=0.25, scale_max=0.5, interpolation=0, always_apply=False, p=1) 不改变图片大小的情况下采样
- **InvertImg** 通过从255减去像素值来反转输入图像。


#### 9.样式变换

- **FDA** (reference_images, beta_limit=0.1, read_fn=<function read_rgb_image at 0x7fefd6731f70>, always_apply=False, p=0.5) 傅立叶域改编 参考图像的文件路径列表或参考图像列表。

#### 10.图像增强

- **FancyPCA** (alpha=0.1, always_apply=False, p=0.5)使用Krizhevsky的论文“具有深度卷积神经网络的ImageNet分类”的FancyPCA增强RGB图像

- **PCA降维过程**：矩阵X[m,n] m条n维数据，维度上求协方差矩阵(维度各自去均值得到X_c*X_c的转置/(n-1))即各维度间的关系系数矩阵[n,n],求出协方差矩阵的特征值和特征向量,特征向量按特征值大小排成矩阵，选前k列[n,k],和原矩阵X相乘得到降维后的矩阵Y[m,k]

- **FancyPCA图像增强过程**：图像[w,h,3]reshape[w*h,3],用PCA的方法得到特征矩阵和特征向量之后,特征向量组成矩阵P[3,3],特征值组成矩阵A[3,1],此时A乘一个数alpha,服从(0,0.1)的高斯分布得到A_a, P*A_a得到一个向量V[3,1],原图的每个像素的三个通道都加上这个向量，得到增强后的图像

PS:感觉是一种对比度的增强 和 均衡直方图应用到图像差不多

#### 11.图像压缩

- **ImageCompression** (quality_lower=99, quality_upper=100, compression_type=<ImageCompressionType.JPEG: 0>, always_apply=False, p=0.5) 

#### 12.其他操作

- **Normalize** (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)

- **Posterize** (num_bits=4, always_apply=False, p=0.5)减少每个颜色通道的位数。
