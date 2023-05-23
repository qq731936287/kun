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


# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

# 训练模型
# 设定模型

model_mlp <- mlp(
  mode = "classification",
  engine = "nnet",
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
)
model_mlp

# workflow
wk_mlp <- 
  workflow() %>%
  add_model(model_mlp) %>%
  add_formula(Lung_mets ~ .)
wk_mlp

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 10)
folds

# 超参数寻优范围
hpset <- parameters(hidden_units(range = c(15, 24)),
                    penalty(range = c(-3, 0)),
                    epochs(range = c(50, 150)))
hpgrid <- grid_regular(hpset, levels = c(5, 2, 2))
hpgrid


# 交叉验证网格搜索过程
set.seed(42)
tune_mlp <- wk_mlp %>%
  tune_grid(resamples = folds,
            grid = hpgrid,
            control = control_grid(save_pred = T, verbose = F))

# 图示交叉验证结果
tune_mlp %>%
  collect_metrics()
autoplot(tune_mlp)

# tune_mlp %>%
#   collect_predictions()

# 经过交叉验证得到的最优超参数
hpbest <- tune_mlp %>%
  select_best(metric = "accuracy")
hpbest

# 采用最优超参数组合训练最终模型
set.seed(42)
final_mlp <- wk_mlp %>%
  finalize_workflow(hpbest) %>%
  fit(traindata2)
final_mlp

# 提取最终的算法模型
final_mlp2 <- final_mlp %>%
  extract_fit_engine()

library(NeuralNetTools)
plotnet(final_mlp2)
garson(final_mlp2) +
  coord_flip()
olden(final_mlp2) +
  coord_flip()
 
# 应用模型-预测训练集
predtrain_mlp <- final_mlp %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
predtrain_mlp
# 评估模型ROC曲线-训练集上
trainroc6 <- predtrain_mlp %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
trainroc6
autoplot(trainroc6)

# 约登法则对应的p值
yueden_mlp <- trainroc6 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_mlp

# 预测概率+约登法则=预测分类
predtrain_mlp2 <- predtrain_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_mlp, "No", "Yes")))
predtrain_mlp2
# 混淆矩阵
cmtrain_mlp <- predtrain_mlp2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_mlp
autoplot(cmtrain_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtrain_mlp %>%
  summary() %>%
  bind_rows(predtrain_mlp %>%
              roc_auc(Lung_mets, .pred_No))
 
# 应用模型-预测测试集
predtest_mlp <- final_mlp %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_mlp
# 评估模型ROC曲线-测试集上
testroc6 <- predtest_mlp %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
testroc6
save(testroc6,file = "testroc6.Rdata")
autoplot(testroc6)


# 预测概率+约登法则=预测分类
predtest_mlp2 <- predtest_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_mlp, "No", "Yes")))
predtest_mlp2
# 混淆矩阵
cmtest_mlp <- predtest_mlp2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_mlp
autoplot(cmtest_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_mlp %>%
  summary() %>%
  bind_rows(predtest_mlp %>%
              roc_auc(Lung_mets, .pred_No))


# 合并训练集和测试集上ROC曲线
roctrain_mlp %>%
  bind_rows(roctest_mlp) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()

#################################################################


# 应用模型-预测测试集
predtest_mlp <- final_mlp %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_mlp
# 评估模型ROC曲线-测试集上
Extestroc6 <- predtest_mlp %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
Extestroc6
autoplot(Extestroc6)
save(Extestroc6,file="Extestroc6.Rdata")

# 预测概率+约登法则=预测分类
predtest_mlp2 <- predtest_mlp %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_mlp, "No", "Yes")))
predtest_mlp2
# 混淆矩阵
cmtest_mlp <- predtest_mlp2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_mlp
autoplot(cmtest_mlp, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_mlp %>%
  summary() %>%
  bind_rows(predtest_mlp %>%
              roc_auc(Lung_mets, .pred_No))

