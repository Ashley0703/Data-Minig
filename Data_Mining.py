# devide the train, valid, and test set  
train_size=0.8  
X = data.drop(columns = ['拘役(天)']).copy()  
y = data['拘役(天)']  
  
# In the first step we will split the data in training and remaining dataset  
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)  
  
# Now since we want the valid and test size to be equal (10% each of overall data).   
# we have to define valid_size=0.5 (that is 50% of remaining data)  
test_size = 0.5  
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)  
  
print(X_train.shape), print(y_train.shape)  
print(X_valid.shape), print(y_valid.shape)  
print(X_test.shape), print(y_test.shape)  
# "Learn" the mean from the training data  
mean_train = np.mean(y_train)  
# Get predictions on the test set  
baseline_predictions = np.ones(y_test.shape) * mean_train  
# Compute MAE  
mae_baseline = mean_absolute_error(y_test, baseline_predictions)  
print("Baseline MAE is {:.2f}".format(mae_baseline))  
from hypopt import GridSearch  
from sklearn.svm import LinearSVC  
param_grid = [  
  {'C': np.logspace(-3, 3, 7), 'loss': ('hinge', 'squared_hinge')}  
 ]  
# Grid-search all parameter combinations using a validation set.  
opt = GridSearch(model = LinearSVC(), param_grid = param_grid)  
opt.fit(X_train, y_train, X_valid, y_valid)  
print('the best patams is{}'.format(opt.get_best_params))  
opt = LinearSVC(C=0.001, class_weight=None, dual=True,  
                           fit_intercept=True, intercept_scaling=1,  
                           loss='squared_hinge', max_iter=1000,  
                           multi_class='ovr', penalty='l2', random_state=0,  
                           tol=0.0001, verbose=0)  
opt.fit(X_train, y_train)  
y_predict = opt.predict(X_test)  
MAE = sum(abs(y_predict - y_test))/len(y_test)  
print('LSVC test 的MAE=',MAE)  
from hypopt import GridSearch  
param_grid = [  
  {'C': np.logspace(-3, 3, 7), 'loss': ('epsilon_insensitive', 'squared_epsilon_insensitive')}  
 ]  
# Grid-search all parameter combinations using a validation set.  
opt = GridSearch(model = LinearSVR(), param_grid = param_grid)  
opt.fit(X_train, y_train, X_valid, y_valid)  
print('the best patams is{}'.format(opt.get_best_params))  
opt = LinearSVR(C=0.001, dual=True, epsilon=0.0, fit_intercept=True,  
                           intercept_scaling=1.0,  
                           loss='squared_epsilon_insensitive', max_iter=1000,  
                           random_state=0, tol=0.0001, verbose=0)  
opt.fit(X_train, y_train)  
y_predict = opt.predict(X_test)  
MAE = sum(abs(y_predict - y_test))/len(y_test)  
print('LSVR test 的MAE=',MAE)  
def commit(province, judge_sex, minority, sex, age, alcohol, confess, admit, imposture, run_away,   
                    highway, accident, injury, full_res, turn_in, reconcile, no_lisence, history):  
    ''''' 
    province, 起诉省份，可输入中文，包括 '江苏省', '江西省', '福建省', '安徽省', '广西壮族自治区',  
    '山东省', '陕西省', '浙江省', '甘肃省','重庆市', '辽宁省', '吉林省', '宁夏回族自治区',  
    '青海省', '湖北省', '内蒙古自治区', '四川省','山西省', '河北省', '广东省', '新疆维吾尔自治区',  
    '海南省', '湖南省', '云南省', '西藏自治区', '黑龙江省', '北京市' 
    judge_sex, 法官性别 0为男 
    minority, 是否少数民族（0-1变量）0为不是 
    sex, 被告性别（0-1变量）0为男 
    age, 被告年龄 int 
    alcohol, 血液酒精浓度，int 
    confess, 是否如实供述（0-1变量）0为不是 
    admit, 是否认罪认罚（0-1变量）0为不是 
    imposture, 是否顶替（0-1变量）0为不是 
    run_away, 是否逃逸（0-1变量）0为不是 
    highway, 是否高速公路酒驾（0-1变量）0为不是 
    accident, 是否造成事故（0-1变量）0为不是 
    injury, 是否造成伤亡（0-1变量）0为不是 
    full_res, 是否全责（0-1变量）0为不是 
    turn_in, 是否自首（0-1变量）0为不是 
    reconcile, 是否谅解（0-1变量）0为不是 
    no_lisence, 是否无证驾驶（0-1变量）0为不是 
    history, 是否有酒驾史 
    '''  
    for province1 in province_map.keys():  
        if province == province1:  
            province = province_map[province1]  
    train = [province, judge_sex, minority, sex, age, alcohol, confess, admit, imposture, run_away,  
                    highway, accident, injury, full_res, turn_in, reconcile, no_lisence, history]  
    train = np.array(train)  
    train = train.reshape(1,-1)  
    return train  
model = xgb.XGBRegressor(max_depth=5, min_child_weight = 4, eta = 0.3, subsample = 0.7, colsample_bytree = 0.7,  
learning_rate=0.1, n_estimators=160, objective='reg:gamma')  
model.fit(X_train,y_train)  
min_mae = float("Inf")  
best_params = None  
for objective in ['reg:squarederror',  'reg:pseudohubererror', 'reg:gamma', 'reg:tweedie']:  
    print("CV with objective={}".format(objective))  
    # We update our parameters  
    params['objective'] = objective  
    # Run and time CV  
    cv_results = xgb.cv(  
        params,  
        dtrain,  
        num_boost_round=num_boost_round,  
        seed=42,  
        nfold=5,  
        metrics=['mae'],  
        early_stopping_rounds=10  
          )  
    # Update best score  
    mean_mae = cv_results['test-mae-mean'].min()  
    boost_rounds = cv_results['test-mae-mean'].argmin()  
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))  
    if mean_mae < min_mae:  
        min_mae = mean_mae  
        best_params = objective  
print("Best params: {}, MAE: {}".format(best_params, min_mae))  
# Define initial best params and MAE  
min_mae = float("Inf")  
best_params = None  
for max_depth, min_child_weight in gridsearch_params:  
    print("CV with max_depth={}, min_child_weight={}".format(  
                             max_depth,  
                             min_child_weight))  
    # Update our parameters  
    params['max_depth'] = max_depth  
    params['min_child_weight'] = min_child_weight  
    # Run CV  
    cv_results = xgb.cv(  
        params,  
        dtrain,  
        num_boost_round=num_boost_round,  
        seed=42,  
        nfold=5,  
        metrics={'mae'},  
        early_stopping_rounds=10  
    )  
    # Update best MAE  
    mean_mae = cv_results['test-mae-mean'].min()  
    boost_rounds = cv_results['test-mae-mean'].argmin()  
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))  
    if mean_mae < min_mae:  
        min_mae = mean_mae  
        best_params = (max_depth,min_child_weight)  
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))  
gridsearch_params = [  
    (subsample, colsample)  
    for subsample in [i/10. for i in range(5,9)]  
    for colsample in [i/10. for i in range(5,9)]  
]  
min_mae = float("Inf")  
best_params = None  
# We start by the largest values and go down to the smallest  
for subsample, colsample in reversed(gridsearch_params):  
    print("CV with subsample={}, colsample={}".format(  
                             subsample,  
                             colsample))  
    # We update our parameters  
    params['subsample'] = subsample  
    params['colsample_bytree'] = colsample  
    # Run CV  
    cv_results = xgb.cv(  
        params,  
        dtrain,  
        num_boost_round=num_boost_round,  
        seed=42,  
        nfold=5,  
        metrics={'mae'},  
        early_stopping_rounds=20  
    )  
    # Update best score  
    mean_mae = cv_results['test-mae-mean'].min()  
    boost_rounds = cv_results['test-mae-mean'].argmin()  
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))  
    if mean_mae < min_mae:  
        min_mae = mean_mae  
        best_params = (subsample,colsample)  
print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))  
min_mae = float("Inf")  
best_params = None  
for eta in [.3, .2, .1, .05, .01, .005]:  
    print("CV with eta={}".format(eta))  
    # We update our parameters  
    params['eta'] = eta  
    # Run and time CV  
    cv_results = xgb.cv(  
        params,  
        dtrain,  
        num_boost_round=num_boost_round,  
        seed=42,  
        nfold=5,  
        metrics=['mae'],  
        early_stopping_rounds=10  
          )  
    # Update best score  
    mean_mae = cv_results['test-mae-mean'].min()  
    boost_rounds = cv_results['test-mae-mean'].argmin()  
    print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))  
    if mean_mae < min_mae:  
        min_mae = mean_mae  
        best_params = eta  
print("Best params: {}, MAE: {}".format(best_params, min_mae))  
best_model = xgb.XGBRegressor(max_depth =  6, min_child_weight = 4, eta = 0.3, subsample = 0.7,  
        colsample_bytree = 0.7, learning_rate = 0.1, n_estimators = 140, objective = 'reg:gamma')  
best_model.fit(X_train,y_train)  
alcohol = range(50,400,20)  
length = []  
for alco in alcohol:  
    result = commit('浙江省', 0, 0, 0, 45, alco, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)  
    y_predict = best_model.predict(result)  
    length.append(int(y_predict))  
fig = plt.figure(figsize=(10,6))  
plt.axis([80, 350, 0, 200])  
plt.xlabel('alcohol')  
plt.ylabel('sentence length')  
plt.plot(alcohol,length)  
plt.scatter(data2['血液乙醇浓度(单位:mg/100ml)'], data2['拘役(天)'],s = 0.2)  
plt.show()  
def gen_predicted_value(predict_value):  
    for i in range(0, len(predict_value)):  
        if predict_value[i] <= 39:  
            predict_value[i] = 30  
        if predict_value[i] <= 41 and predict_value[i] > 39:  
            predict_value[i] = 40  
        if predict_value[i] <= 48 and predict_value[i] > 41:  
            predict_value[i] = 45  
        if predict_value[i] <= 52.5 and predict_value[i] > 48:  
            predict_value[i] = 50  
        if predict_value[i] <= 66 and predict_value[i] > 52.5:  
            predict_value[i] = 60  
        if predict_value[i] <= 84 and predict_value[i] > 66:  
            predict_value[i] = 75  
        if predict_value[i] <= 105 and predict_value[i] > 84:  
            predict_value[i] = 90  
        if predict_value[i] > 105:  
            predict_value[i] = 120  
    return predict_value  
alcohol = range(50,350,5)  
length = []  
for alco in alcohol:  
    result = commit('浙江省', random.randint(0,1), random.randint(0,1), random.randint(0,1), 45, alco, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, random.randint(0,1))  
    y_predict = best_model.predict(result)  
    length.append(gen_predicted_value(y_predict))  
fig = plt.figure(figsize=(10,6))  
plt.axis([80, 350, 0, 200])  
plt.xlabel('alcohol')  
plt.ylabel('sentence length')  
plt.scatter(alcohol,length, s=20, c = '#8c564b')  
plt.scatter(data2['血液乙醇浓度(单位:mg/100ml)'], data2['拘役(天)'],s = 0.2)  
plt.show()  
