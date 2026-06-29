from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import HuberRegressor,LinearRegression,Ridge,Lasso,RidgeCV
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_absolute_percentage_error
import pandas as pd
import joblib
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class AbstractionModels:
    def __init__(self, random_state=42) -> None:
        self.random_state = random_state  # 保存为实例变量
        self.models = {
            'xgboost': XGBRegressor(random_state=self.random_state, n_jobs=1),
            'lightgbm': LGBMRegressor(random_state=self.random_state, objective='regression', n_jobs=1),
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=1),
            'adaboost': AdaBoostRegressor(random_state=self.random_state),
            'catboost': CatBoostRegressor(random_state=self.random_state, verbose=0,  thread_count=1),
            'svm': SVR(),
            'gbdt': GradientBoostingRegressor(random_state=self.random_state),
            'cart': DecisionTreeRegressor(random_state=self.random_state),
            'plsr': PLSRegression(),
            'mlp': MLPRegressor(random_state=self.random_state)
        }
        self.param_grid = {
            'xgboost': {
                # 树的数量
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
                # 树的最大深度
                'max_depth': [2, 3, 4, 5, 7, 9],
                # 学习率
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.8, 1.0],              # 可选：样本采样比例
                'colsample_bytree': [0.8, 1.0]        # 可选：特征采样比例
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [5, 10],
                'num_leaves': [31, 63],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'min_child_samples': [20],
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],              # 随机森林需要较多树
                # 回归任务可深一些
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                # 叶子节点最小样本数
                'min_samples_leaf': [1, 2,],
                'max_features': ['sqrt', 'log2', 0.5],   # 特征采样比例
                # 是否bootstrap
                'bootstrap': [True, False]
            },
            'adaboost': {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                # 回归任务的损失函数
                'loss': ['linear', 'square', 'exponential']
                # 注意：AdaBoost回归使用 AdaBoostRegressor，loss参数很重要
            },
            'catboost': {
                'iterations': [100, 150, 200],           # 迭代次数
                'learning_rate': [0.05, 0.1],
                # 树深度
                'depth': [10, 20, 30],
                # L2正则化（回归重要）
                'l2_leaf_reg': [1, 3, 5],
                # 特征分箱
                'border_count': [64, 128, 256],
                'bagging_temperature': [0.5],             # 随机采样
                # 随机性强度
                'random_strength': [0.5],
                # 叶子值估计方法
                'leaf_estimation_method': ['Newton', 'Gradient']
            },
            'gbdt': {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'cart': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [1, 2, 5],
                'min_samples_leaf': [12, 24, 48],
                'max_features': [None, 'sqrt', 'log2']
            },
            'svm': {
                'kernel': ['rbf'],               # rbf通常效果最好
                'C': [1],                # 正则化参数（回归重要）
                'gamma': ['auto'],   # 核函数系数
                # SVR的epsilon管（不敏感损失）
                'epsilon': [0.01],
                # 仅对poly核有效
                'degree': [2]
            },
            'plsr': {
                'n_components': [1,2,3],
                'scale': [True, False],  # 是否标准化
                'max_iter': [100, 200, 500]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate_init': [0.001, 0.01, 0.05, 0.1],
                'max_iter': [100, 200, 500],
                'batch_size': [16, 32, 64],
                'solver': ['adam', 'sgd'],
                'momentum': [0.9, 0.95]  # 仅当 solver='sgd' 时有效
            }
        }

    def create_model(self):
        pass

    def fit(self):
        pass

    def trainTestSplits(self):
        pass

    def save_bset_param(self):
        pass

    def save_best_model(self):
        pass

    def save_pri_information(self):
        pass


class BaseModel(AbstractionModels):
    def __init__(self,
                 model_name='xgboost',
                 random_state=42,
                 data_file='/share/home/jinhm/PhD_Project/B7_WAVE_U/DataSet.xlsx',
                 output_file='/share/home/jinhm/PhD_Project/B7_WAVE_U/result',
                 feature_cols=['u10', 'cp', 'kp','Hs', 'wave_steepnessdou', 'B*'],
                 target_col='u_star_down',
                 test_size=0.25
                 ) -> None:
        super(BaseModel, self).__init__(random_state=random_state)
        self.model_name = model_name
        self.output_file = output_file
        self.random_state = random_state

        self.model = self.create_model()
        self.X_train, self.X_test, self.y_train, self.y_test = self.trainTestSplits(
            data_file, feature_cols, target_col, test_size, random_state)

    def create_model(self):
        self.model = self.models[self.model_name]
        print(f'实例化了一个模型：{self.model_name}!')
        return self.model

    def fit(self):
        # 更新您的 scoring 字典
        scoring = {
            'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                                greater_is_better=False),  # ✅ 新增 RMSE
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            'r2': make_scorer(r2_score, greater_is_better=True)  # R2 越大越好
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid[self.model_name],
            cv=kf,
            scoring=scoring,
            refit='r2',
            verbose=1,
            n_jobs=16,
            return_train_score=True
        )

        grid_search.fit(self.X_train, self.y_train)

        self.save_bset_param(grid_search)
        self.save_best_model(grid_search)
        self.save_pri_information(grid_search)

    def trainTestSplits(self, data_file, feature_cols, target_col, test_size, random_state):
        data = pd.read_excel(data_file)

        X = data[feature_cols]
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def save_bset_param(self, grid_search):
        # 构建路径
        dir_path = os.path.join(self.output_file, f'{self.model_name}')
        file_path = os.path.join(dir_path, 'best_params.pkl')

        # 创建文件夹（关键）
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(grid_search.best_params_, file_path)
        print(grid_search.best_params_)

    def save_best_model(self, grid_search):
        # 构建路径
        dir_path = os.path.join(self.output_file, f'{self.model_name}')
        file_path = os.path.join(dir_path, 'best_model.pkl')

        # 创建文件夹（关键）
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(grid_search.best_estimator_, file_path)

    def save_pri_information(self, grid_search):
        # 1. 最佳参数组合的各个指标（平均值）
        best_idx = grid_search.best_index_

        # 验证
        best_test_rmse = -grid_search.cv_results_['mean_test_rmse'][best_idx]
        best_test_rmse_std = grid_search.cv_results_['std_test_rmse'][best_idx]

        best_test_mae = -grid_search.cv_results_['mean_test_mae'][best_idx]
        best_test_mae_std = grid_search.cv_results_['std_test_mae'][best_idx]

        best_test_r2 = grid_search.cv_results_['mean_test_r2'][best_idx]
        best_test_r2_std = grid_search.cv_results_['std_test_r2'][best_idx]

        best_test_mape = -grid_search.cv_results_['mean_test_mape'][best_idx] * 100
        best_test_mape_std = grid_search.cv_results_['std_test_mape'][best_idx] * 100

        # 训练
        best_train_rmse = -grid_search.cv_results_['mean_train_rmse'][best_idx]
        best_train_rmse_std = grid_search.cv_results_['std_train_rmse'][best_idx]

        best_train_mae = -grid_search.cv_results_['mean_train_mae'][best_idx]
        best_train_mae_std = grid_search.cv_results_['std_train_mae'][best_idx]

        best_train_r2 = grid_search.cv_results_['mean_train_r2'][best_idx]
        best_train_r2_std = grid_search.cv_results_['std_train_r2'][best_idx]

        best_train_mape = -grid_search.cv_results_['mean_train_mape'][best_idx] * 100
        best_train_mape_std = grid_search.cv_results_['std_train_mape'][best_idx] * 100
        # ========== 保存为 CSV ==========
        result = [
            ['test',str(grid_search.best_params_),best_test_rmse,best_test_rmse_std,best_test_mae,best_test_mae_std,best_test_r2,best_test_r2_std,best_test_mape,best_test_mape_std],
            ['train',str(grid_search.best_params_),best_train_rmse,best_train_rmse_std,best_train_mae,best_train_mae_std,best_train_r2,best_train_r2_std,best_train_mape,best_train_mape_std]
        ]
        # 创建 DataFrame
        columns = ['dataset_type', 'best_params', 'rmse', 'rmse_std', 'mae', 'mae_std', 
           'r2', 'r2_std', 'mape', 'mape_std']

        best_results_df = pd.DataFrame(result, columns=columns)

        # 构建路径
        dir_path = os.path.join(self.output_file, f'{self.model_name}')
        file_path = os.path.join(dir_path, 'Metric.csv')

        # 创建文件夹（关键）
        os.makedirs(dir_path, exist_ok=True)
        best_results_df.to_csv(file_path, index=False)

class MetaModel:
    def __init__(   self,
                    data_file='/share/home/jinhm/PhD_Project/B7_WAVE_U/DataSet.xlsx',
                    feature_cols=['u10','Hs', 'direction','wave_steepnessdou', 'B*', 'cp', 'kp', 'fp_comp'],
                    target_col='u_star_down',
                    test_size=0.25,
                    model_path='/share/home/jinhm/PhD_Project/B7_WAVE_U/result',
                    random_state=42
                ):
        self.models_name_dict = {
                                'xgboost':'XGBoost',
                                'lightgbm':'LightGBM',
                                'random_forest':'Random Forest',
                                'adaboost':'AdaBoost',
                                'catboost':'CatBoost',
                                'svm':'SVM',
                                'gbdt':'GBDT',
                                'cart':'CART',
                                'plsr':'PLSR',
                                'mlp':'MLP'
                            }
        self.random_state = random_state
        self.model_path = model_path
        self.data_file = data_file
        self.feature_cols =feature_cols
        self.target_col = target_col
        self.test_size = test_size
        
    def fit(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        data = pd.read_excel(self.data_file)
        X = data[self.feature_cols]
        y = data[self.target_col]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 训练并验证模型性能
        tr_metric = []
        models = []
        for train_idx, val_idx in kf.split(X_train):
            _, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            _, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            en_train = []
            for name, _ in self.models_name_dict.items():
                model_path = f'{self.model_path}/{name}/best_model.pkl'
                model = joblib.load(model_path)
                en_train.append(model.predict(X_val))
            # ✅ 按列拼接（关键）
            en_train = np.column_stack(en_train)

            # 实例化模型
            # meta_model = HuberRegressor(epsilon=1000, alpha=0.01)
            # meta_model = LinearRegression()
            # meta_model = Ridge(alpha=0.2)
            meta_model = LGBMRegressor()
            # 保存模型
            meta_model.fit(en_train,y_val)
            models.append(meta_model)
            # 预测模型
            en_tr_pred = meta_model.predict(en_train)
            # ===== 3. 计算指标 =====
            train_r2 = r2_score(y_val, en_tr_pred)
            train_rmse = np.sqrt(mean_squared_error(y_val, en_tr_pred))
            train_mape = mean_absolute_percentage_error(y_val, en_tr_pred) * 100
                
            tr_metric.append([train_r2, train_rmse, train_mape])

        tr_metric = np.vstack(tr_metric)

        # R2 绝对最优模型
        id = np.argmax(tr_metric[:,0])
        best_model = models[id]
        self.save_best_model(best_model)
        self.save_pri_information(tr_metric)


    def save_best_model(self, model):
        # 构建路径
        dir_path = os.path.join(self.model_path, f'HuberRegressor')
        file_path = os.path.join(dir_path, 'best_model.pkl')

        # 创建文件夹（关键）
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(model, file_path)
    
    def save_pri_information(self,tr_metric):
        print(tr_metric)
        mean_tr_metric = np.mean(tr_metric,axis=0)
        std_tr_metric = np.std(tr_metric,axis=0)

        # 假设指标名称（按你的实际情况改）
        metric_names = ['R2', 'RMSE', 'MAPE']

        # 构建 DataFrame
        df = pd.DataFrame({
            'Metric': metric_names,
            'Train Mean': mean_tr_metric,
            'Train Std': std_tr_metric
        })

        # 保存为 CSV
        dir_path = os.path.join(self.model_path, f'HuberRegressor')
        file_path = os.path.join(dir_path, 'Metric.csv')

        # 创建文件夹（关键）
        os.makedirs(dir_path, exist_ok=True)
        df.to_csv(file_path, index=False)

if __name__ == '__main__':
    # level 0 train
    # parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=str, default='svm')
    # parser.add_argument('target', type=str, default='u_star_down')
    # args = parser.parse_args()

    # # 最优结果
    # # levelmodel = BaseModel(args.model, feature_cols=['u10','Hs', 'direction','wave_steepnessdou', 'B*', 'cp', 'kp', 'fp_comp'], target_col=args.target)
    # levelmodel = BaseModel(args.model, feature_cols=['u10','Hs', 'direction','wave_steepnessdou', 'B*','z/l'], target_col=args.target)
    # model = levelmodel.fit()

    # level 1 train
    model = MetaModel(feature_cols=['u10','Hs', 'direction','wave_steepnessdou', 'B*','z/l'])
    model.fit()