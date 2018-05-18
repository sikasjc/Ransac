# RANSAC
RANSAC is the abbreviation of Random Sample Consensus. 

It is a general algorithm that can be used with other parameter estimation methods in order to obtain robust models with a certain probability when the noise in the data doesn't obey the general noise assumption.
More Information about Ransac can see [Ransac-tutorial][1].

## Ransac
`Ransac.py` implements the basic Ransac algorithm to fit a line model. The absolute distance is used in the distance function. The fitting function uses the `np.linalg.lstsq` least square method and the testransac.py test algorithm output.
This [post][2](Chinese) has more information about code.

Ransac.py实现了基本的Ransac算法，来拟合一个直线模型。在距离函数上使用绝对距离，拟合函数利用np.linalg.lstsq最小二乘法，testransac.py测试算法输出。
这篇[文章][3]写了我对RANSAC的理解和python代码。

**output**

    ============== Paramters ===============  
    [0.5, -2.9994712862791754e-15]
    =============== Residual ===============
    1.0658141036401503e-14
    ============== Iteration ===============  
    404 iteration finds the best params
    ================= Time =================
    3.3 msecs mean time spent per iteration

![ransac][4]

## Ransac with Adaptive Parameter Calculation
The main idea of Adaptive Parameter Calculation is calculating the proportion of inliers(w) on run time and modifying the other parameters according to the new value of w. It can be used not only for complexity reduction but also when there is no information about w for the data set.

Adaptive Parameter Calculation starts with a worst case value of w and reassigns the better value when it founds a consensus set whose proportion of data samples is higher than current w. And then it updates number of iterations(k)and minimum consensus threshold(d) with the new
value of w.

This [post][5](Chinese) has more information about code.

自适应参数计算的主要思想是在运行时，计算内群（inliers）$w$ 的比例，然后根据新的 $w$ 的值修改其他参数。它不仅可以用于降低计算的复杂性，而且可以用于当数据集没有关于 $w$ 的信息时。

自适应参数计算从最坏的情况开始计算 $w$，当找到一致集（consensus set）的数据样本在整个数据集中的比例高于当前 ww 时，重新分配更好的值给 $w$。之后用新的 $w$ 来更新迭代次数 $k$ 和最小一致阈值 $d$ 。

更多请看此：[here][6]

**output**

    ============== Paramters =============== 
    [0.5000000000000001, -1.4098251542937202e-15]
    =============== Residual ===============
    1.135175287103607e-14
    ============== Iteration ===============  
    113 the number of iterations
    ================= Time =================
    2.4 msecs mean time spent per iteration

![ransac_APC][7]

## Vanishing point detection using Ransac
More information about the code: [here][8](Chinese)

基本流程如下：

 1. 建立一个拟合模型，选择拟合函数和距离函数。这里，拟合函数选择将多条线，按顺序两两寻找它们的相交点，最后取这些相交点的平均位置为我们估计的消失点位置。距离函数选择点到直线的正交距离。
 2. 从线段数据集中随机采样，也为内群 (inliers)，因为两条线相交于一点，这里我们可以选择随机采样数为2。
 3. 根据随机采样的线段（数目为2），估计其相交点，即消失点的位置。 
 4. 计算估计的消失点到线段数据集中其他线段的距离，距离小于阈值的线段加入内群(inliers) 。
 5. 再次估计内群中的线段相交点位置，同时如第1点所述，计算模型的拟合误差。
 6. 不断重复1~5，最终选择模型拟合误差最小的消失点估计位置

更多请看此：[here][9]
![lane][10]
![edge][11]
 **output**
 ![point][12]


  [1]: http://www.math-info.univ-paris5.fr/~lomn/Cours/CV/SeqVideo/Material/RANSAC-tutorial.pdf
  [2]: https://sikasjc.github.io/2018/04/21/Ransac/
  [3]: https://sikasjc.github.io/2018/04/21/Ransac/
  [4]: http://ormnbkvfv.bkt.clouddn.com/18-4-22/92143570.jpg
  [5]: https://sikasjc.github.io/2018/05/04/APC_ransac/
  [6]: https://sikasjc.github.io/2018/05/04/APC_ransac/
  [7]: http://ormnbkvfv.bkt.clouddn.com/18-5-4/67603747.jpg
  [8]: https://sikasjc.github.io/2018/04/27/vanishing_point/
  [9]: https://sikasjc.github.io/2018/04/27/vanishing_point/
  [10]: http://ormnbkvfv.bkt.clouddn.com/18-4-27/30300732.jpg
  [11]: http://ormnbkvfv.bkt.clouddn.com/18-4-27/24445890.jpg
  [12]: http://ormnbkvfv.bkt.clouddn.com/18-4-27/53704287.jpg
