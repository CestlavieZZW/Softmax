# Softmax
main与main10CROSS为两段主函数 main实现的是数据集和测试集64分的验证，并测试最佳的正则化参数。 main实现按照10折交叉验证，测试最佳的正则化参数。 divide函数是实现随机数据划分。 cross1_10可以实现任意比例数据交叉验证划分，ave函数实现平均概率。
