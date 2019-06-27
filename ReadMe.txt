本文件夹包含MODEL文件，里面是模型和训练好的参数，由于今早不小心把训练好的model的参数给替换了
，所以临时训练了个，训练时间可能不是很够，只能凑合用了。
模型采用resnet的整体设计框架，激活函数为prelu,每一层卷积之后都有BN层。采用的正则化技术是参考了biggan的思想，用的正交正则化。。。。

运行
直接bash run,sh即可生成answer文件，修改文件路径请在src/下的ECG_predict中修改BASE_DIR即可。
MODEL中的src里面存放着训练参数以及predict，network,train等文件。
手动程序预测的话，只需要运行ECG_predict.py文件就可以了

可能训练来不及了 ，  直接进入995-icu那个文件夹，修改src下的ECG_predicting中的BASE_DIR，右键运行程序即可，就会重新生成answer文件
