# 1.封装函数file_name打开的文件名
# data是csv倒入时的数据集¶
# data_rfe在后面会有，是rfe特征选择后的总数据集
# s_rfe 是rfe特征选择后的特征数据
# target是目标数据
#1.1打开csv并存到data中
import numpy as np
def file_name(name):
    import pandas as pd
    global data
    data = pd.read_csv(name)
    print(data)
#1.2画所有列分布的柱状图
def hist():
    import matplotlib.pyplot as plt
    # 绘制柱状图，其中bins设置为50
    data.hist(bins=50, figsize=(20,15))
    plt.tight_layout()
    plt.savefig('./hist_allFeatures.png', dpi=300, bbox_inches = 'tight')
    plt.show()
#2.封装函数特征选择之前heatmap画热图
def heatmap_before():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    featureData=data.iloc[:,:]
    global corMat
    corMat = pd.DataFrame(featureData.corr())  #corr 求相关系数矩阵
    corMat.to_csv('./heatmap-before.csv')
    plt.figure(figsize=(20, 30))
    sns.heatmap(corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
    plt.savefig('./heatmap-before.png', dpi=300, bbox_inches = 'tight')
    plt.show()
    return
#3. rfe特征选择 feature_rfe_select1 is easier
def feature_rfe_select1(remain_number):
    # 使用随机森林的rfe:RandomForestRegressor()
    from sklearn import preprocessing
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestRegressor
    # 输入数据归一化
    X = data.values[:, :-1]
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    y = data.values[:, -1]  # 目标数值
    # rfe步骤
    model = RandomForestRegressor()
    rfe = RFE(estimator=model, n_features_to_select=remain_number, step=1)
    rfe_X = rfe.fit_transform(X, y)
    print("特征是否被选中：\n", rfe.support_)
    print("获取的数据特征尺寸:", rfe_X.shape)
    # 打印rfe后的特征，但可能包含空值
    import pandas as pd
    Features_0 = pd.DataFrame(data=data.iloc[:, :-1].columns, columns=['Features'])
    Features_0
    Features_rfe = pd.DataFrame(data=rfe.support_, columns=['whether selected'])
    Features_rfe
    #     pd.options.display.max_rows=None
    p = pd.concat([Features_0, Features_rfe], axis=1)
    q = p[p['whether selected']>0]
    r = q.reset_index(drop=True)
    global s_rfe
    s_rfe = pd.DataFrame(data=data,columns=r.Features.values)
    global target
    target = pd.DataFrame(data=data.iloc[:,-1])
    # target = pd.DataFrame(data, columns=['Potential (v)'])
    global data_rfe
    data_rfe = pd.concat([s_rfe,target], axis=1)
    print("最后的特征s_rfe:", r.Features.values)
    print("目标target:", target)
    print("rfe后的总数据data_rfe:", data_rfe)
#4.1 画rfe特征选择后的热图
def heatmap_afterRFE():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    data_rfe_corMat = pd.DataFrame(data_rfe.corr())  #corr 求相关系数矩阵
    data_rfe_corMat.to_csv('./heatmap-afterRFE.csv')
    plt.figure(figsize=(20, 30))
    sns.heatmap(data_rfe_corMat, annot=False, vmax=1, square=True, cmap="Blues",linewidths=0)
    plt.savefig('./heatmap-afterRFE.png', dpi=300, bbox_inches = 'tight')
    plt.show()
#4.2 画rfe特征选择后的pairplot图
def pairplot_afterRFE():
    import seaborn as sns
    import matplotlib.pyplot as plt
    g6 = sns.pairplot(data_rfe, kind='reg')
    plt.savefig('./sns-pairplot-remain.png', dpi=300, bbox_inches = 'tight')
    plt.show()
#5 重要性排名（皮尔逊系数）
#5.1 特征选择之前所有特征的重要性
def FeatureImportance_before(rotationDeg,fontsize_axis,figure_size_xaxis,figure_size_yaxis):
    import pandas as pd
    FirstLine=corMat.iloc[-1,:]
    FirstLine=pd.DataFrame(FirstLine)
    FirstLine_Del_Target=FirstLine.iloc[:-1,:]
    importance=FirstLine_Del_Target.sort_values(by=FirstLine_Del_Target.columns.tolist()[-1],ascending=False)
    # importance=FirstLine_Del_Target.sort_values(by="Potential (v)",ascending=False)
    print(importance)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Times New Roman']
    # plt.rcParams ['font.sans-serif'] ='SimHei'    #显示中文
    plt.rcParams ['axes.unicode_minus']=False    #显示负号
    importance.plot(kind='bar', figsize=(figure_size_xaxis,figure_size_yaxis), rot=rotationDeg, fontsize=8)  #colormap='rainbow'
    plt.savefig('./FeatureImportance_before.png', dpi=300, bbox_inches = 'tight')
    plt.show()
#5.2 特征选择之后的个别特征的重要性
def FeatureImportance_afterRFE(rotationDeg, fontsize_axis, figure_size_xaxis, figure_size_yaxis):
    import pandas as pd
    corMat_rfe = pd.DataFrame(data_rfe.corr())  # corr 求相关系数矩阵
    FirstLine_rfe = corMat_rfe.iloc[-1, :]
    FirstLine_rfe = pd.DataFrame(FirstLine_rfe)
    FirstLine_rfe_Del_Target = FirstLine_rfe.iloc[:-1, :]
    # importance_rfe = FirstLine_rfe_Del_Target.sort_values(by="Potential (v)", ascending=False)
    importance_rfe = FirstLine_rfe_Del_Target.sort_values(by=FirstLine_rfe_Del_Target.columns.tolist()[-1],ascending=False)
    print(importance_rfe)
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    importance_rfe.plot(kind='bar', figsize=(figure_size_xaxis, figure_size_yaxis), rot=rotationDeg,
                        fontsize=8)  # colormap='rainbow'
    plt.savefig('./FeatureImportance_after.png', dpi=300, bbox_inches='tight')
    plt.show()
#6 机器学习建模
# 6.1.1 xgboost默认超参数建模画图
# (n_estimators=2000, max_depth=100, eta=0.1, gamma=0,
# subsample=0.9, colsample_bytree=0.9, learning_rate=0.2)
def xgboost_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt#计算准确率xgboost
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #xgboost建模
    from xgboost import XGBRegressor
    clf = XGBRegressor(n_estimators=2000, max_depth=100, eta=0.1, gamma=0,
                       subsample=0.9, colsample_bytree=0.9, learning_rate=0.2)
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('xgboost-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:",scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i+1,"scores_mean:",scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('xgboost-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train_prediction, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('xgboost-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
# 6.1.2 xgboost自己修改超参数, 建模
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)
def xgboost_modify(a, b, c, d, e, f, g):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # xgboost建模
    from xgboost import XGBRegressor
    clf = XGBRegressor(n_estimators=a, max_depth=b, eta=c, gamma=d,
                       subsample=e, colsample_bytree=f, learning_rate=g)
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig('xgboost-modify.tif', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:",scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i+1,"scores_mean:",scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('xgboost_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train_prediction, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('xgboost-modify-train-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
# 6.1.3 xgboost randomSearchCV, 包含了交叉验证
def xgboost_RandomSearchCV():
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 尝试random search
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBRegressor
    param_distribs = {
        'n_estimators': range(80, 200, 40),
        'max_depth': range(2, 15, 4),
        'learning_rate': np.linspace(0.01, 2, 4),
        'subsample': np.linspace(0.7, 0.9, 4),
        'colsample_bytree': np.linspace(0.5, 0.98, 4),
        'min_child_weight': range(1, 9, 3)
    }
    #         "n_estimators": [100, 200, 300, 400, 500],
    #         'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'learning_rate': [0.01, 0.05, 0.1],
    #         'subsample': [0.6, 0.8, 1.0],
    #         'colsample_bytree': [0.6, 0.8, 1.0],
    #         'min_child_weight': [1, 5, 10],
    #         'gamma': [0.5, 1, 1.5, 2, 5],
    #         "objective": "multi:softmax",
    #         "aplha": [0, 2, 4, 6, 8]
    clf = XGBRegressor()
    global rnd_search_cv_xgboost
    rnd_search_cv_xgboost = RandomizedSearchCV(clf, param_distribs, n_iter=300, cv=10, scoring='neg_mean_squared_error')
    rnd_search_cv_xgboost.fit(X_train, y_train)
    y_prediction = rnd_search_cv_xgboost.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig('xgboost-RandomizedSearchCV.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(rnd_search_cv_xgboost, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    #     ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('Xgboost_rnd_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
   # 训练集也可以打印准确率并plot图
    y_train_prediction = rnd_search_cv_xgboost.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('xgboost-train-randomSearch.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.1.4 xgboost GridSearchCV网格搜索（不随机）, 包含了交叉验证
def xgboost_GridSearchCV():
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt  # 计算准确率xgboost
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 尝试random search
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBRegressor
    param_distribs = {
        'n_estimators': range(80, 200, 30),
        'max_depth': range(2, 15, 3),
        'learning_rate': np.linspace(0.01, 2, 4),
        'subsample': np.linspace(0.7, 0.9, 4),
        'colsample_bytree': np.linspace(0.5, 0.98, 4),
        'min_child_weight': range(1, 9, 3)
    }
    clf = XGBRegressor()
    grid_search_cv = GridSearchCV(clf, param_distribs, cv=10, scoring='neg_mean_squared_error')
    grid_search_cv.fit(X_train, y_train)
    y_prediction = grid_search_cv.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig('xgboost-GridSearchCV.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(grid_search_cv, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('grid_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
   # 训练集也可以打印准确率并plot图
    y_train_prediction = grid_search_cv.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('Xgboost-grid_search_train', dpi=300, bbox_inches = 'tight')
    plt.show()

#6.2 随机森林机器学习建模
# 6.2.1 随机森林默认超参数建模画图
def RandomForest_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])

    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #Random forest建模
    from sklearn import ensemble
    clf = ensemble.RandomForestRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('randomForest-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    #     xminorLocator   = MultipleLocator(1000)
    #     yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    #     plt.xlim(1.5,9.5)
    plt.ylim(0, 1.2)
    #     plt.minorticks_on()
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('randomForest-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('randomForest-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.2.2 Random forest modify 自己修改超参数, 建模
def RandomForest_modify(a, b, c, d, e):
# max_depth, max_features, min_samples_split, n_estimators, random_state
# 20, 0.3, 2, 10, 10
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # RandomForest建模
    from sklearn import ensemble
    clf = ensemble.RandomForestRegressor(max_depth=a,max_features=b, min_samples_split=c,n_estimators=d,random_state=e)
    clf.fit(X_train, y_train)
    y_prediction = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig('RandomForest-modify.tif', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('RandomForest_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('RandomForest-modify-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.2.3 RandomForest randomSearchCV, 包含了交叉验证
def RandomForest_RandomSearchCV():
    # 数据切分
    import numpy as np
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 尝试random search
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn import ensemble
    param_distribs = {'bootstrap': [True, False],
               'max_depth': [10, 20, 50, 100, 110],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [1, 2, 4],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [130, 180, 230]}
    clf = ensemble.RandomForestRegressor()
    rnd_search_cv = RandomizedSearchCV(clf, param_distribs, n_iter=300, cv=10, scoring='neg_mean_squared_error')
    rnd_search_cv.fit(X_train, y_train)
    y_prediction = rnd_search_cv.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1 / 2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:", rmse)
    print("MAE:", MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:", R2)
    print("MSE:", MSE)
    # plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator = MultipleLocator(1000)
    yminorLocator = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties='Times New Roman', size=20)
    plt.ylabel("Prediction", fontproperties='Times New Roman', size=20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties='Times New Roman',
             size=20, horizontalalignment='center')
    plt.savefig('RandomForest-RandomizedSearchCV.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(rnd_search_cv, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('RandomForest_rnd_search_cv-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
   # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('RandomForest-train-randomSearchCV.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

#6.3 bagging机器学习建模
# 6.3.1 bagging默认超参数建模画图
from sklearn import ensemble
def Bagging_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.BaggingRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('Bagging-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('Bagging-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('Bagging-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

#6.4 AdaBoost机器学习建模
# 6.3.1 AdaBoost默认超参数建模画图
from sklearn import ensemble
def AdaBoost_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.AdaBoostRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('AdaBoost-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('AdaBoost-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('AdaBoost-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

#6.5 GradientBoosting机器学习建模
# 6.6.1 GradientBoosting默认超参数建模画图
from sklearn import ensemble
def GradientBoosting_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 机器学习建模
    from sklearn import ensemble
    clf = ensemble.GradientBoostingRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('GradientBoosting-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('GradientBoosting-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('GradientBoosting-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

#6.6 ExtraTree机器学习建模
# 6.6.1 ExtraTree默认超参数建模画图
def ExtraTree_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.tree import ExtraTreeRegressor
    clf = ExtraTreeRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('ExtraTree-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('ExtraTree-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('ExtraTree-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.7 svm机器学习建模
# 6.7.1 svm默认超参数建模画图
def svm_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import svm
    clf = svm.SVR()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('svm-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('svm-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('svm-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.8 DecisionTree机器学习建模
# 6.8.1 DecisionTree默认超参数建模画图
def DecisionTree_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn import tree
    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('DecisionTree-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('DecisionTree-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('DecisionTree-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.9 LinearRegression机器学习建模
# 6.9.1 LinearRegression默认超参数建模画图
def LinearRegression_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import pandas as pd
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('LinearRegression-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('LinearRegression-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('LinearRegression-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.10 Ridge机器学习建模
# 6.10.1 Ridge默认超参数建模画图
def Ridge_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.linear_model import Ridge
    clf = Ridge()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('Ridge-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('Ridge-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('Ridge-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.11 MLP机器学习建模
# 6.11.1 MLP默认超参数建模画图
def MLP_default():
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor()
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('MLP-default.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('MLP-default-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('MLP-default-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

# 6.11.2 MLP_modify手动修改超参数建模画图
def MLP_modify(l,a,m,ha,hb):
    from sklearn import preprocessing
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    # 数据切分
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    #机器学习建模
    from sklearn.neural_network import MLPRegressor
    clf = MLPRegressor(solver='lbfgs', activation='relu', learning_rate_init=l, alpha=a, max_iter=m,
                 hidden_layer_sizes=(ha, hb))
    clf.fit(X_train, y_train)
    y_prediction=clf.predict(X_test)
    # 打印准确率
    mse = mean_squared_error(y_test, y_prediction)
    rmse = mse ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE = mean_absolute_error(y_test, y_prediction)
    print("RMSE:",rmse)
    print("MAE:",MAE)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2 = r2_score(y_test, y_prediction)
    MSE = mean_squared_error(y_test, y_prediction)
    print("R2:",R2)
    print("MSE:",MSE)
    #plot图
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_test, y_test, label='Real Data')
    plt.scatter(y_test, y_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE, MSE, R2), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('MLP_modify.tif', dpi=300, bbox_inches = 'tight')
    plt.show()
    # 使用KFold交叉验证建模
    from sklearn.model_selection import cross_val_score
    kfold = KFold(n_splits=10)
    scores = cross_val_score(clf, X_train, y_train, scoring='r2', cv=kfold)
    # scoring='neg_mean_squared_error'
    print("scores:", scores)
    scores_fold = []
    for i in range(len(scores)):
        scores_mean = scores[:i + 1].mean()
        print(i + 1, "scores_mean:", scores_mean)
        scores_fold.append(scores_mean)
    # 使用KFold交叉验证plot图
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(range(1, 11), scores_fold, c='r')
    plt.scatter(range(1, 11), scores_fold, c='r')
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4, width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    x_major_locator = MultipleLocator(1)  # 把x轴的刻度间隔设置为1，并存在变量里
    ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
    y_major_locator = MultipleLocator(0.2)  # 把y轴的刻度间隔设置为10，并存在变量里
    ax.yaxis.set_major_locator(y_major_locator)  # 把y轴的主刻度设置为10的倍数
    plt.ylim(0, 1.2)
    plt.xlabel("k", fontproperties='Times New Roman', size=24)
    plt.ylabel("score", fontproperties='Times New Roman', size=24)
    plt.savefig('MLP_modify-10-fold-crossvalidation.png', dpi=300, bbox_inches='tight')
    plt.show()
    # 训练集也可以打印准确率并plot图
    y_train_prediction = clf.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_prediction)
    rmse_train = mse_train ** (1/2)
    from sklearn.metrics import mean_absolute_error
    MAE_train = mean_absolute_error(y_train, y_train_prediction)
    print("RMSE:", rmse_train)
    print("MAE:", MAE_train)
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    R2_train = r2_score(y_train, y_train_prediction)
    MSE_train = mean_squared_error(y_train, y_train_prediction)
    print("R2:",R2_train)
    print("MSE:",MSE_train)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.rcParams['font.sans-serif'] = 'Roman'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.plot(y_train, y_train, label='Real Data')
    plt.scatter(y_train, y_train_prediction, label='Predict', c='r')
    ax=plt.gca()
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)
    plt.tick_params(width=2)
    ax.xaxis.set_tick_params(labelsize=24)
    plt.tick_params(which='major',length=8)
    plt.tick_params(which='minor',length=4,width=2)
    ax.yaxis.set_tick_params(labelsize=24)
    xminorLocator   = MultipleLocator(1000)
    yminorLocator   = MultipleLocator(1000)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    plt.minorticks_on()
    plt.xlabel("True", fontproperties = 'Times New Roman', size = 20)
    plt.ylabel("Prediction", fontproperties = 'Times New Roman', size = 20)
    plt.text(.05, .2, 'MAE = %.3f \nMSE =  %.3f \nR2 =  %.3f \n' % (MAE_train, MSE_train, R2_train), fontproperties = 'Times New Roman', size = 20, horizontalalignment='center')
    plt.savefig('MLP_modify-train.tif', dpi=300, bbox_inches = 'tight')
    plt.show()


# 7.1.1 预测集基于xgboost_default()
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)
def xgboost_default_predict(csvName):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # xgboost建模
    from xgboost import XGBRegressor
    clf = XGBRegressor()
    clf.fit(X_train, y_train)
    # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
    import pandas as pd
    x_New = pd.read_csv(csvName)
    print("new features dataset: ", x_New)
    # xgboost_modify新的预测
    y_New_prediction = clf.predict(x_New)
    y_New_prediction = pd.DataFrame(y_New_prediction)
    y_New_prediction.columns = ['Output']
    print("new output: ", y_New_prediction)
    NewData = pd.concat([x_New, y_New_prediction], axis=1)
    print("New total Data: ", NewData)
    NewData.to_csv("New_prediction_total_xgboost_default.csv")

# 7.1.2 预测集基于xgboost_modify
# 画图得到拟合图以及交叉验证图
# (n_estimators=2000xxx, max_depth=100xxx, eta=0.1xxx, gamma=0xxx,
# subsample=0.9xxx, colsample_bytree=0.9xxx, learning_rate=0.2xxx)
def xgboost_modify_predict(a, b, c, d, e, f, g, csvName):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # xgboost建模
    from xgboost import XGBRegressor
    clf = XGBRegressor(n_estimators=a, max_depth=b, eta=c, gamma=d,
                       subsample=e, colsample_bytree=f, learning_rate=g)
    clf.fit(X_train, y_train)
    # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
    import pandas as pd
    x_New = pd.read_csv(csvName)
    print("new features dataset: ", x_New)
    # xgboost_modify新的预测
    y_New_prediction = clf.predict(x_New)
    y_New_prediction = pd.DataFrame(y_New_prediction)
    y_New_prediction.columns = ['Output']
    print("new output: ", y_New_prediction)
    NewData = pd.concat([x_New, y_New_prediction], axis=1)
    print("New total Data: ", NewData)
    NewData.to_csv("New_prediction_total_xgboost_modify.csv")

# 7.1.3 预测集基于rnd_search_cv_xgboost
# 画图得到拟合图以及交叉验证图
def rnd_search_cv_xgboost_predict(csvName):
    # 数据切分
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    X = s_rfe
    y = target
    X = X.values[:, :]
    y = y.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    # 数据归一化
    for i in range(X_train.shape[1]):
        X_train[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_train[:, [i]])
    for i in range(X_test.shape[1]):
        X_test[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X_test[:, [i]])
    # 需要准备新的待预测的特征集x_New.csv(不含目标列), 导入 x_New的列数为之前设置的rfe剩余特征个数
    import pandas as pd
    x_New = pd.read_csv(csvName)
    print("new features dataset: ", x_New)
    # xgboost_modify新的预测
    y_New_prediction = rnd_search_cv_xgboost.predict(x_New)
    y_New_prediction = pd.DataFrame(y_New_prediction)
    y_New_prediction.columns = ['Output']
    print("new output: ", y_New_prediction)
    NewData = pd.concat([x_New, y_New_prediction], axis=1)
    print("New total Data: ", NewData)
    NewData.to_csv("New_prediction_total_rnd_search_cv_xgboost.csv")


# 8 描述符导入
# 8.1 有机分子描述符导入（NJmatML提供了pydel描述符和rdkit描述符）
# 8.1.1 pydel描述符
# 8.1.1.1 导入有机分子smiles码的csv文件


def smiles_csv_pydel(name2):
    import pandas as pd
    global data2
    data2 = pd.read_csv(name2)
    print(data2.iloc[:,0])

# 8.1.1.2 pydel描述符生成
def pydel_featurizer():
    from padelpy import from_smiles
    import pandas as pd
    data2a = data2.iloc[:,0].map(lambda x : from_smiles(x).values())
    data2a = pd.DataFrame(data2a)
    data2b = data2a.iloc[:,0].apply(pd.Series)
    #写入列名
    data2c = data2.iloc[:,0].map(lambda x : from_smiles(x).keys())
    col2c = data2c.iloc[0]
    data2b.columns = col2c
    print(data2b)
    # 特征存入pydel_featurizer.csv
    data2b.to_csv("pydel_featurizer_output.csv")

# !pip install padelpy
# from padelpy import from_smiles
# import pandas as pd
# # calculate molecular descriptors for propane
# CCC_descriptors = from_smiles('CCC')
# print(CCC_descriptors)
# print(CCC_descriptors['nAcid'])
# print(CCC_descriptors['ALogP'])
# print(CCC_descriptors['ALogp2'])


# 8.1.2 rdkit描述符
# 8.1.2.1 导入有机分子smiles码的csv文件
def smiles_csv_rdkit(name3):
    import pandas as pd
    global data3
    data3 = pd.read_csv(name3)
    print(data3.iloc[:,0])



# 8.1.2.2 rdkit描述符生成
def rdkit_featurizer():
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    # choose 200 molecular descriptors
    chosen_descriptors = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    # create molecular descriptor calculator
    mol_descriptor_calculator = MolecularDescriptorCalculator(chosen_descriptors)
    data4 = data3.iloc[:,0].map(lambda x : mol_descriptor_calculator.CalcDescriptors(Chem.MolFromSmiles(x)))
    data4 = pd.DataFrame(data4)
    data5 = pd.DataFrame()
    # split to 200 columns
    for i in range(0, 200):
        data5 = pd.concat([data5, data4.applymap(lambda x: x[i])], axis=1)
    data5.columns = chosen_descriptors
    print(data5)
    # 特征存入rdkit_featurizer.csv
    data5.to_csv("rdkit_featurizer_output.csv")

# 8.1.2.3 从smiles码画分子
def drawMolecule(smiles):
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import rdDepictor
    from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
    m = Chem.MolFromSmiles(smiles)
    display(m)

# 8.1.2.3 从smiles码画分子
# drawMolecule('Cc1ccccc1') #括号里（SMILES码两边）请加引号


# 8.2 无机材料描述符 (NJmatML参考Matminer使用类独热编码方式特征化无机化学式)
# 8.2.1 导入含有无机材料化学式的csv
def inorganic_csv(name4):
    import pandas as pd
    global data4
    data4 = pd.read_csv(name4)
    print(data4)


# 8.2.2 matminer无机材料（类独热编码）描述符生成，102维
# 例如(Fe2AgCu2)O3, Fe2O3, Cs3PbI3, MoS2, CuInGaSe, Si, TiO2等
def inorganic_featurizer():
    import pandas as pd
    from matminer.featurizers.composition.element import ElementFraction
    from pymatgen.core import Composition
    ef = ElementFraction()
    list4 = list(map(lambda x: Composition(x), data4.iloc[:,0]))
    data7 = pd.DataFrame()
    for i in range(0, len(data4.index)):
        data7 = pd.concat([data7, pd.DataFrame(ef.featurize(list4[i])).T])
    data8 = data7.reset_index()
    print(data8)
    element_fraction_labels = ef.feature_labels()
    print(element_fraction_labels)
    # 特征存入pydel_featurizer.csv
    data8.to_csv("inorganic_featurizer_output.csv")


# 9 遗传算法设计新特征
## 9.1 普通默认运算符
def gp_default(r_thresh):  ## 输入参数为皮尔森阈值 ：例如输入0.6后，大于0.6的才显示
    import numpy as np
    from sklearn import preprocessing
    from gplearn import genetic
    X = data.values[:,:-1]
    y = data.values[:,-1]
    for i in range(X.shape[1]):
        X[:,[i]] = preprocessing.MinMaxScaler().fit_transform(X[:,[i]])
    est_gp = genetic.SymbolicTransformer(population_size=1000,
                               generations=91, stopping_criteria=0.01,
                               p_crossover=0.8, p_subtree_mutation=0.05,
                               p_hoist_mutation=0.05, p_point_mutation=0.05,
                               max_samples=0.9, verbose=1,
                               parsimony_coefficient=0.01, random_state=None,n_components=100)
    V=est_gp.fit(X, y)
    px=V.transform(X)
    for i in range(0,50):
      pear=np.corrcoef(px[:,i], y)
      pea=pear[0,1]
      if pea>r_thresh:
       print(pea,i,data.columns.values.tolist()[i], V[i])
    print('\n***************************')
    for i in range(len(data.columns.values.tolist())):
       print(i, data.columns.values.tolist()[i])

## 9.2 更多运算符
def gp_tan(r_thresh):
    import numpy as np
    from sklearn import preprocessing
    from gplearn import genetic
    X = data.values[:, :-1]
    y = data.values[:, -1]
    for i in range(X.shape[1]):
        X[:, [i]] = preprocessing.MinMaxScaler().fit_transform(X[:, [i]])
    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs', 'neg','inv','sin','cos','tan', 'max', 'min']
    est_gp = genetic.SymbolicTransformer(population_size=1000,
                               generations=91, stopping_criteria=0.01,
                               p_crossover=0.8, p_subtree_mutation=0.05,
                               p_hoist_mutation=0.05, p_point_mutation=0.05,
                               max_samples=0.9, verbose=1,function_set=function_set,
                               parsimony_coefficient=0.01, random_state=None,n_components=100)
    V=est_gp.fit(X, y)
    px=V.transform(X)
    for i in range(0,50):
      pear=np.corrcoef(px[:,i], y)
      pea=pear[0,1]
      if pea>r_thresh:
       print(pea,i, V[i])
    print('\n***************************')
    for i in range(len(data.columns.values.tolist())):
       print(i, data.columns.values.tolist()[i])


## 9.3 tSR默认形式为(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
def tSR_default(r_thresh):
    import numpy as np
    X = data.values[:, :-1]
    y = data.values[:, -1]
    for i in range(0,(data.shape[1]-1)):
     for j in range(0,(data.shape[1]-1)):
      for k in range(0,(data.shape[1]-1)):
        for n in range(0,(data.shape[1]-1)):
         px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
         per=np.corrcoef(px, y)
         if per[0,1]>r_thresh or per[0,1]< -1 * r_thresh:
          print(per[0,1])
          print(i,j,k,n)
          print(data.columns.values.tolist()[i],data.columns.values.tolist()[j],data.columns.values.tolist()[k],data.columns.values.tolist()[n])
          print('(',data.columns.values.tolist()[i],'-',data.columns.values.tolist()[j],')','*','(',data.columns.values.tolist()[k],'-',data.columns.values.tolist()[n],')')
          print('**********************************************')

# ## 9.4 tSR更多运算符，默认形式为(X[:,i]-X[:,j])*(X[:,k]-X[:,n]) 可能删去，没有用处
# def tSR_tan(r_thresh):
#     import numpy as np
#     X = data.values[:, :-1]
#     y = data.values[:, -1]
#     for i in range(0,(data.shape[1]-1)):
#      for j in range(0,(data.shape[1]-1)):
#       for k in range(0,(data.shape[1]-1)):
#         for n in range(0,(data.shape[1]-1)):
#          px=(X[:,i]-X[:,j])*(X[:,k]-X[:,n])
#          per=np.corrcoef(px, y)
#          if per[0,1]>r_thresh or per[0,1] < -1 * r_thresh:
#           print(per[0,1])
#           print(i,j,k,n)
#           print(data.columns.values.tolist()[i],data.columns.values.tolist()[j],data.columns.values.tolist()[k],data.columns.values.tolist()[n])
#           print('(',data.columns.values.tolist()[i],'-',data.columns.values.tolist()[j],')','*','(',data.columns.values.tolist()[k],'-',data.columns.values.tolist()[n],')')
#           print('**********************************************')

#未完待续（其他机器学习算法，网格搜索，预测集建立，描述符填充等等）