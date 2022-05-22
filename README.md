# License-Plate-Recognition
 Computer Vision Course Project



目  录

 

第一章 绪论----------------------------------------------------------------------------------------------------1

1.1 实验要求分析--------------------------------------------------------------------------------1

第二章 车牌位置检测算法开发---------------------------------------------------------------------1

2.1 预处理----------------------------------------------------------------------------------------------1

2.2 边缘检测&轮廓提取--------------------------------------------------------------------------------1

2.3 基于车牌颜色进行定位----------------------------------------------------------------------------2

2.4 矩形矫正-------------------------------------------------------------------------------------------3

第三章 文字识别算法的开发-------------------------------------------------------------------------------4

3.1 字符分割----------------------------------------------------------------------------------------------4

3.2 SVM模型训练----------------------------------------------------------------------------------------5

3.3 字符识别----------------------------------------------------------------------------------------------5

第四章 可视化界面开发-------------------------------------------------------------------------------------5

第五章 算法性能评估-------------------------------------------------------------------------------------5

5.1 输入任务图片进行测试---------------------------------------------------------------------------6

5.1.1 测试过程---------------------------------------------------------------------------------------6

5.1.2 实验结果分析-------------------------------------------------------------------------------10

5.2 更大范围的测试---------------------------------------------------------------------------11

5.2.1 测试过程-------------------------------------------------------------------------------------11

5.2.2 实验结果分析-------------------------------------------------------------------------------23

第六章 总结与回顾-----------------------------------------------------------------------------------24

参考文献------------------------------------------------------------------------------------------------------25



 

第一章 **绪论**

 

1.1 实验要求分析

本次实验要求设计算法检测车牌位置并识别车牌号，通过恰当的方式对检测过程以及识别结果进行可视化，并设计一些指标来评价算法的性能。因此，实验将在以下方面开展：

（1）车牌位置检测算法的开发

（2）文字识别算法的开发

（3）可视化界面开发

（4）在召回率、准确率、执行速度三个层面对算法性能进行评估

 

第二章 ***\*车牌\*******\*位置检测算法开发\****

 

2.1 预处理

由于图像质量容易受光照、天气、相机位置等因素的影响，所以在识别车牌之前需要先对图像做一些预处理，以保证得到车牌最易识别的图像。一般会对图像进行噪声过滤、对比度增强、图像缩放等处理。去噪方法有均值滤波、中值滤波和高斯滤波等；增强对比度的方法有对比度线性拉伸、直方图均衡和同态滤波器等；图像缩放的主要方法有最近邻插值法、双线性插值法和立方卷积插值等。在opencv中，有一些现成的函数可以调用，例如cv2.GaussianBlur、cv2.resize、cv.medianBlur等，极大的减小了工作量。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps3.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps4.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps5.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps6.jpg)

2.2 边缘检测&轮廓提取

数字图像的边缘检测是图像分割、目标区域的识别、区域形状提取等图像分析领域十分重要的基础。所谓边缘是指其周围像素灰度值有阶跃变化或屋顶变化的那些像素点的集合。边缘广泛存在于物体与背景之间、物体与物体之间、图像基元与基元之间。它是图像分割所依赖的重要特征，图像理解和分析的第一步一般都是边缘检测。边缘检测的实质是采用某种算法来提取出图像中对象与背景间的交界线。我们将边缘定义为图像中灰度发生急剧变化的区域边界。图像灰度的变化情况可以用图像灰度分布的梯度来反映，因此我们可以用局部图像微分技术来获得边缘检测算子。

本项目中，我们首先使用cv2.cvtColor函数获得灰度图像，然后使用cv2.threshold函数进行二值化处理，最后使用cv2.Canny函数获得图像边缘。![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps7.png)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps8.jpg)

完成边缘检测后，我们通过findContours函数寻找可能存在车牌的矩形区域并且面积和长宽比筛选不符合条件的矩形。

2.3 基于车牌颜色进行定位

国内的民用车牌只有蓝、绿、黄这三个特殊颜色，因此，我们也可以考虑基于车牌的颜色特征进行定位。我们先规定好颜色阈值，然后调用cv2.inRange函数去除背景，并使用cv2.morphologyEx()函数去除边缘毛刺，最终截取出车牌图像。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps9.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps10.jpg) 

基于颜色定位时，回报率比基于边缘检测高，但由于更易受环境光照影响，图像截取精度不稳定，导致最终识别精度降低。本项目中，我们同时采取两种方案互为补充。

2.4 矩形矫正

本是矩形的车牌畸变后变成了平行四边形，因此车牌轮廓和得出来的矩形轮廓并不契合。但有了矩形的四个顶点坐标后，可以通过简单的几何相似关系求出平行四边形车牌的四个顶点坐标。

　　平行四边形四个顶点与矩形四个顶点之间有如下关系：矩形顶点Top_Point、Bottom_Point与平行四边形顶点new_top_point、new_bottom_point重合，矩形顶点Top_Point的横坐标与平行四边形顶点new_right_point的横坐标相同，矩形顶点Bottom_Point的横坐标与平行四边形顶点new_left_point的横坐标相同。

但事实上，由于拍摄的角度不同，可能出现两种不同的畸变情况。可以根据矩形倾斜角度的不同来判断具体是哪种畸变情况。判断出具体的畸变情况后，选用对应的几何相似关系，即可轻易地求出平行四边形四个顶点坐标，即得到了畸变后车牌四个顶点的坐标。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps11.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps12.jpg) 

第三章 文字识别算法的开发

 

3.1 字符分割

依据车牌特点可知，每个字符之间都存在一定的纯黑区域。二值化后，字符区域为白色，车牌中的非字符区域为黑色。

首先计算每列中白色像素点个数，垂直投影后得到直方图，并以此来判断各个字符的起始位置。然后从左至右扫描投影直方图，找到存在白色像素点的第一列，则认定为该列是车牌第一个字符的左边界。若上一列存在白色像素点，而下一列是不存在白色像素点的黑色区域，则认定该列为第一个字符的右边界，同理，便可分割出其余6个车牌字符左右边界。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps13.jpg) 

3.2 SVM模型训练

我最初准备直接使用Tesseract-OCR 引擎的pytesseract工具，但经过测试后发现效果并不理想，经常识别不出结果。车牌采用的字体特殊，并且提取出的图像难免存在畸变、模糊，直接采用现成的OCR工具并不现实。

 SVM实质上是一个类分类器，是一个能够将不同类样本在样本空间分隔的超平面。换句话说，给定一些标记的训练样本，SVM算法输出一个最优化的分隔超平面。

我们先生成一个SVM模型，设置核函数及SVM模型类型。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps14.jpg) 

 

然后调用方法，并分别饲喂中文数据、字符数据。（若已经训练好模型就执行IF语句中的load操作（调取持久化模型），否则要是没有模型就开始训练）。我们采用os.walk方法，遍历一个目录内各个子目录和子文件。os.path.basename(),返回path最后的文件名，将标记好的训练集喂给分类器。训练完成后，数据存储在svm.dat、svmchinese.dat文件中。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps15.jpg)![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps16.jpg) 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps17.jpg) 

3.3 字符识别

基于已经完成分割的图像块，我们进行最终的识别。依据车牌特点可知，第一个字符为汉字，我们调用modelchinese.predict识别汉字，后面的字符使用model.predict识别。

使用if np.mean(part_card) < 255 / 5筛查可能被误识别的铆钉；以字符粗细及顺序为标准，筛查可能被误识别为1的边缘。

最终，得到识别结果。

 

***\*第四章 可视化界面开发\****

采用Python 的标准 GUI 库Tkinter，开发了一个简单的ui界面，可以同时展示基于颜色和基于边缘检测的识别结果。这一部分与计算机视觉关系不大，不详细叙述。

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps18.jpg) 

***\*第五章\**** ***\*算法性能评估\****

5.1 输入任务图片进行测试

5.1.1 

插入以下代码以统计时间：

time_start = time.perf_counter()  # 记录开始时间

\# function()  执行的程序

time_end = time.perf_counter()  # 记录结束时间

time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s

print(time_sum)

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps19.jpg) 

1-1识别结果正确 运行时间：0.11457489998429082

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps20.jpg) 

1-2 识别结果正确 运行时间：0.3740752000012435

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps21.jpg) 

1-3识别结果正确 运行时间：0.16662090001045726

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps22.jpg) 

2-1 识别结果正确 运行时间：0.634753999998793

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps23.jpg) 

2-2 识别结果正确 运行时间：1.1005535000003874

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps24.jpg) 

2-3 识别结果正确 运行时间：0.6740618999756407

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps25.jpg) 

3-1 识别失败

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps26.jpg) 

3-2 图像截取正确，识别结果不准确 运行时间：0.8194302000047173

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps27.jpg) 

3-3 识别失败

 

5.1.2 实验结果分析

对于简单、中等难度的图片，算法能高效的识别，而对于偏转角度较大的图片，算法识别的准确率明显降低。算法的运行时间与识别难度正相关，较复杂的图片耗时更长，但总体都在可接受的范围内。对于这个较小的测试集，召回率为77.77%，准确率为85.7%。其中颜色定位召回率为77.77%，准确率为85.7%；而形状定位召回率为22.22%，准确率为100%。对于这个较小的测试集，颜色定位召回率较高，准确率稍低；形状定位召回率不理想，但准确率高。

 

 

 

 

 

 

 

 

 

5.2 更大范围的测试

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps28.jpg) 

（1）识别准确 运行时间0.14867019999655895 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps29.jpg) 

（2）识别准确 运行时间0.11569109998526983

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps30.jpg) 

（3）形状定位识别准确，颜色定位识别失败 运行时间0.3593366999994032

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps31.jpg) 

（4）颜色定位识别准确，形状定位识别错误（裁切过多） 运行时间1.3363534999953117

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps32.jpg) 

（5）识别准确 运行时间0.08527509999112226

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps33.jpg) 

（6）形状定位识别准确，颜色定位识别错误 运行时间0.2379265000054147

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps34.jpg) 

（7）形状定位识别准确，颜色定位识别错误 运行时间1.0948923000250943

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps35.jpg) 

（8）识别准确 运行时间0.2333327999804169

 

 

 

 

 

 

 

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps36.jpg) 

（9）形状定位识别准确，颜色定位识别失败 运行时间0.1605555000132881

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps37.jpg) 

（10）形状定位识别准确，颜色定位识别失败 运行时间0.140456000022823

 

 

 

 

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps38.jpg) 

（11）识别成功 运行时间0.17137549998005852

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps39.jpg) 

（12）颜色定位识别准确，形状定位识别失败 运行时间1.3363534999953117

 

 

 

 

 

 

 

 

 

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps40.jpg) 

（13）识别成功 运行时间0.6755698000197299

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps41.jpg) 

（14）识别成功 运行时间0.5767195000080392

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps42.jpg) 

（15）识别成功 运行时间0.06534370000008494

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps43.jpg) 

（16）形状定位识别错误，颜色定位识别错误 运行时间0.21858469999278896

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps44.jpg) 

（17）识别成功 运行时间0.10229410001193173

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps45.jpg) 

（18）形状定位识别成功，颜色定位识别失败 运行时间0.1401320000004489

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps46.jpg) 

（19）形状定位识别错误，颜色定位识别成功 运行时间0.2296511000022292

 

 

 

 

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps47.jpg) 

（20）识别成功 运行时间0.12802229999215342

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps48.jpg) 

（21）识别成功 运行时间0.6581497000006493

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps49.jpg) 

（22）识别成功 运行时间0.06670689999009483

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps50.jpg) 

（23）识别成功 运行时间0.11089810001431033

 

 

 

 

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps51.jpg) 

（24）形状定位识别错误，颜色定位识别成功 运行时间0.2447229000099469

 

![img](file:///C:\Users\future\AppData\Local\Temp\ksohtml40868\wps52.jpg) 

（25）识别成功 运行时间1.2389736000041012

 

5.2.2 测试结果分析

平均运行时间：0.4431749194775s，速度较快

总准确率：96%

总召回率：100%

形状定位准确率：75%

形状定位召回率：96%

颜色定位准确率：85.71%

颜色定位召回率：84%

颜色定位易受光照等环境因素影响，因此在本次实验中，召回率稍低，因此规避开了一些难度较大的图像，反而获得了更高的准确率。而形状定位稳定性更好，召回率更高，但在极端条件下会识别出错，导致准确率降低。两种方式混合使用，召回率与准确率都得到了有效的提高。

**第****六****章** **总结与回顾**

 

车牌识别算法在当下应用广泛，是智慧城市建设的重要基础设施。车牌识别基于图像分割和图像识别理论，对含有车牌识别车辆号牌的图像进行分析处理，从而确定牌照在图像中的位置，并进一步提取和识别出文本字符。车牌识别过程包括图像采集、预处理、车牌定位、字符分割、字符识别、结果输出等一系列算法运算。

从车辆照片输入，到最终结果输出，其中包含的图像处理、目标检测、文字识别，刚好大致涵盖了本学期课程的大致脉络，通过这次实践，我对本课程知识有了更进一步的理解。在实际工程中，我遇到了各种各样的问题，在解决这些问题的过程中，我查阅资料、优化代码的能力得到了极大的提高。

感谢老师及助教学长学姐的付出，谢谢你们带我走进计算机视觉的世界。

参考文献

 

[1][基于色彩的车牌识别研究](https://kns.cnki.net/kcms/detail/detail.aspx?filename=XDJS202032013&dbcode=CJFQ&dbname=CJFDTEMP&v=TvxEf3bHZ8GL50Dv1tfaPGTsg63PPKCwD6_ypt27aHSk0MJFDtxBQpaKQuSg0_Ye)[J]. 唐愉顺,张生果,牛潞,崔泽宇. 现代计算机. 2020(32)

[2][SVM与MLP在车牌识别中的应用研究](https://kns.cnki.net/kcms/detail/detail.aspx?filename=LZKQ202005008&dbcode=CJFQ&dbname=CJFDTEMP&v=vtHpRCJf9_tma7W2-R9Jgk_KPR6N743FO226Wd1hAxSU4niWZfMWNmPrnGVi1Qiz)[J]. 张晨. 甘肃科技纵横. 2020(05)

[3][车牌识别中二值化方法的研究](https://kns.cnki.net/kcms/detail/detail.aspx?filename=JYRJ200702012&dbcode=CJFQ&dbname=cjfd2007&v=LIRxg4SuVTzcZ5aKre7fMuIho8sibKZgpgZyaT44ZM2P-9Jv0CN8Dd505_MEPuRe)[J]. 朱浩悦,耿国华,周明全. 计算机应用与软件. 2007(02)

[4]汽车车牌识别系统实现（三）-- 车牌矫正+字符分割+代码实现[OL]. 2020-10-25 https://blog.csdn.net/dongjinkun/article/details/105429239

[5]Python OCR工具pytesseract详解[OL].测试开发小记,2021-12-21

https://blog.csdn.net/u010698107/article/details/121736386

[6]车牌识别项目（2）膨胀与腐蚀方案[OL].[Coin_Anthony](https://blog.csdn.net/m0_37921318) 2020-05-10

https://blog.csdn.net/m0_37921318/article/details/106044679

[7] [图像处理] Python+OpenCV实现车牌区域识别及Sobel算子丨[OL].eastmount.2021/10/03

https://bbs.huaweicloud.com/blogs/detail/302933

[8] 基于图像处理的车牌识别系统[OL]. 数据建模案例2020-09-20

https://zhuanlan.zhihu.com/p/256403571

[9] 图像识别原理简介——以车牌识别为例[OL]. 詹姆士2021-01-17

https://zhuanlan.zhihu.com/p/344984911

[10]计算机图像处理技术在车牌识别系统中的应用[J]. 张旭东.  科学大众(科学教育). 2014(12)

[11]实战：基于OpenCV 的车牌识别[OL].小白学视觉.2021-03-28

https://blog.csdn.net/qq_42722197/article/details/115291543

[12]最大稳定极值区域（MSER）检测[OL].zizi7.2015-12-22

https://blog.csdn.net/zizi7/article/details/50379973

[13]Python Tkinter 框架控件（Frame）[OL].Python GUI编程

https://www.runoob.com/python/python-tk-frame.html

[16]CarPlateIdentity[OL]simple2048.31Dec 2019.https://github.com/simple2048/CarPlateIdentity

[14]不同程度倾斜下的车牌定位和车牌矫正方法[J]. 陆华章,温浩.  现代工业经济和信息化. 	2016(05)

[15]基于灰度边缘和车牌颜色对的车牌定位[J]. 王钰淞.  信息与电脑(理论版). 2012(02)

## [16]基于深度学习高性能中文车牌识别 High Performance Chinese License Plate RecognitionFramework.[OL].[szad670401](https://github.com/szad670401)/[HyperLPR](https://github.com/szad670401/HyperLPR)https://github.com/szad670401/HyperLPR

 

 

 

 

 

 

 

 
