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


datarecipe <- recipe(Lung_mets ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  prep()
datarecipe


# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL)
testdata2 <- bake(datarecipe, new_data = testdata)
test2 <- bake(datarecipe, new_data = test)

# 数据预处理
# 先对照训练集写配方
colnames(train)
datarecipe <- recipe(Lung_mets ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = traindata)
testdata2 <- bake(datarecipe, new_data = testdata)
test2 <- bake(datarecipe, new_data = test)

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

# 训练模型
# 设定模型

model_rsvm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_rsvm

# workflow
wk_rsvm <- 
  workflow() %>%
  add_model(model_rsvm) %>%
  add_formula(Lung_mets ~ .)
wk_rsvm

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 10)
folds

# 超参数寻优范围
hpset <- parameters(cost(range = c(-5, 5)), 
                    rbf_sigma(range = c(-4, -1)))
hpgrid <- grid_regular(hpset, levels = c(2,3))
hpgrid


# 交叉验证网格搜索过程
tune_rsvm <- wk_rsvm %>%
  tune_grid(resamples = folds,
            grid = hpgrid,
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
tune_rsvm %>%
  collect_metrics()
autoplot(tune_rsvm)


# 经过交叉验证得到的最优超参数
hpbest <- tune_rsvm %>%
  select_best(metric = "accuracy")
hpbest

# 采用最优超参数组合训练最终模型
final_rsvm <- wk_rsvm %>%
  finalize_workflow(hpbest) %>%
  fit(traindata2)
final_rsvm

# 提取最终的算法模型
final_rsvm %>%
  extract_fit_engine()


# 应用模型-预测训练集
predtrain_rsvm <- final_rsvm %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
train$prob_svm=predtrain_rsvm$.pred_Yes
# 评估模型ROC曲线-训练集上
trainroc3 <- predtrain_rsvm %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
trainroc3

# 约登法则对应的p值
yueden_rsvm <- trainroc3 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rsvm
# 预测概率+约登法则=预测分类
predtrain_rsvm2 <- predtrain_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rsvm, "No", "Yes")))
predtrain_rsvm2
# 混淆矩阵
cmtrain_rsvm <- predtrain_rsvm2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_rsvm
autoplot(cmtrain_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtrain_rsvm %>%
  summary() %>%
  bind_rows(predtrain_rsvm %>%
              roc_auc(Lung_mets, .pred_No))


# 应用模型-预测测试集
predtest_rsvm <- final_rsvm %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_rsvm
# 评估模型ROC曲线-测试集上
testroc3 <- predtest_rsvm %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(testroc3)

# 预测概率+约登法则=预测分类
predtest_rsvm2 <- predtest_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rsvm, "No", "Yes")))
predtest_rsvm2
# 混淆矩阵
cmtest_rsvm <- predtest_rsvm2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_rsvm
autoplot(cmtest_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_rsvm %>%
  summary() %>%
  bind_rows(predtest_rsvm %>%
              roc_auc(Lung_mets, .pred_No))

# 合并训练集和测试集上ROC曲线
roctrain_rsvm %>%
  bind_rows(roctest_rsvm) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

#############################################################

# 应用模型-预测测试集
predtest_rsvm <- final_rsvm %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_rsvm

# 评估模型ROC曲线-测试集上
Extestroc3 <- predtest_rsvm %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(Extestroc3)

# 预测概率+约登法则=预测分类
predtest_rsvm2 <- predtest_rsvm %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rsvm, "No", "Yes")))
predtest_rsvm2
# 混淆矩阵
cmtest_rsvm <- predtest_rsvm2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_rsvm
autoplot(cmtest_rsvm, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
cmtest_rsvm %>%
  summary() %>%
  bind_rows(predtest_rsvm %>%
              roc_auc(Lung_mets, .pred_No))


table(traindata$Lung_mets)



library(pmsampsize)
pmsampsize( 
  type="b", #b对应binary二分类型
  rsquared = 0.05, #预期的R2cs值
  parameters = 8, #候选预测参数的数目
  prevalence = 0.05, #结局事件发生的总体比例
  seed = 123456 #设置的随机种子
)


