# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/rand_forest.html
load("traindata.Rdata")
load("testdata.Rdata")
load("test.Rdata")
library(tidymodels)
library(tidyverse)

for(i in c(1,3:5,7:9)){
  traindata[[i]] <- factor(traindata[[i]])
}
for(i in c(1,3:5,7:9)){
  testdata[[i]] <- factor(testdata[[i]])
}
for(i in c(1,3:5,7:9)){
  test[[i]] <- factor(test[[i]])
}



# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(Lung_mets ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()
datarecipe
 

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL)
testdata2 <- bake(datarecipe, new_data = testdata)
test2 <- bake(datarecipe, new_data = test)



# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)
skimr::skim(test2)

# 训练模型
# 设定模型

model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2)
model_xgboost

# workflow
wk_xgboost <- 
  workflow() %>%
  add_model(model_xgboost) %>%
  add_formula(Lung_mets ~ .)
wk_xgboost

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 10)
folds

# 超参数寻优范围
hpset <- parameters(
  mtry(range = c(2, 8)),
  min_n(range = c(5, 20)),
  tree_depth(range = c(1, 3)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0)),
  sample_prop(range = c(0.8, 1))
)
set.seed(42)
# grid_regular()
hpgrid <- grid_random(
  hpset, size = 10
)
hpgrid


# 交叉验证随机搜索过程
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_grid(resamples = folds,
            grid = hpgrid,
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
tuneresult <- tune_xgboost %>%
  collect_metrics()
autoplot(tune_xgboost)

# 经过交叉验证得到的最优超参数
hpbest <- tune_xgboost %>%
  select_best(metric = "accuracy")
hpbest

# 采用最优超参数组合训练最终模型
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest) %>%
  fit(traindata2)
final_xgboost


# 提取最终的算法模型
final_xgboost2 <- final_xgboost %>%
  extract_fit_engine()

importance_matrix <- xgb.importance(model = final_xgboost2)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix,
                    measure = "Cover",
                    col = "skyblue")
# SHAP
xgb.plot.shap(data = as.matrix(traindata2[,-7]), 
              model = final_xgboost2,
              top_n = 5)

library(SHAPforxgboost)
shap <- shap.prep(final_xgboost2, 
                  X_train = as.matrix(traindata2[,-7]))
shap.plot.summary(shap) +
  labs(title = "SHAP for XGBoost") +
  theme(axis.text.y = element_text(size = 15))

load("")
# 应用模型-预测训练集
predtrain_xgboost <- final_xgboost %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
predtrain_xgboost
# 评估模型ROC曲线-训练集上
trainroc4 <- predtrain_xgboost %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
trainroc4
autoplot(trainroc4)
 
# 约登法则对应的p值
yueden_xgboost <- trainroc4 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_xgboost
# 预测概率+约登法则=预测分类
predtrain_xgboost2 <- predtrain_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_xgboost, "No", "Yes")))
# 混淆矩阵
cmtrain_xgboost <- predtrain_xgboost2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_xgboost
autoplot(cmtrain_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtrain_xgboost %>%
  summary() %>%
  bind_rows(predtrain_xgboost %>%
              roc_auc(Lung_mets, .pred_No))

# 应用模型-预测测试集
predtest_xgboost <- final_xgboost %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_xgboost
# 评估模型ROC曲线-测试集上
testroc4 <- predtest_xgboost %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "testdata")
autoplot(testroc4)


# 预测概率+约登法则=预测分类
predtest_xgboost2 <- predtest_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_xgboost, "No", "Yes")))
# 混淆矩阵
cmtest_xgboost <- predtest_xgboost2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_xgboost
autoplot(cmtest_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_xgboost %>%
  summary() %>%
  bind_rows(predtest_xgboost %>%
              roc_auc(Lung_mets, .pred_No))


# 合并训练集和测试集上ROC曲线
roctrain_xgboost %>%
  bind_rows(roctest_xgboost) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()






 
colnames(testdata)
# 应用模型-预测测试集
predtest_xgboost <- final_xgboost %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_xgboost
# 评估模型ROC曲线-测试集上
Extestroc4 <- predtest_xgboost %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(Extestroc4)


# 预测概率+约登法则=预测分类
predtest_xgboost2 <- predtest_xgboost %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_xgboost, "No", "Yes")))
# 混淆矩阵
cmtest_xgboost <- predtest_xgboost2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_xgboost
autoplot(cmtest_xgboost, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_xgboost %>%
  summary() %>%
  bind_rows(predtest_xgboost %>%
              roc_auc(Lung_mets, .pred_No))


