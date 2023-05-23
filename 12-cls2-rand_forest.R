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


# 训练模型
# 设定模型

model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf

# workflow
wk_rf <- 
  workflow() %>%
  add_model(model_rf) %>%
  add_formula(Lung_mets ~ .)
wk_rf

# 重抽样设定-5折交叉验证
set.seed(32)
folds <- vfold_cv(traindata2, v = 10)
folds

# 超参数寻优范围
hpset <- parameters(
  mtry(range = c(2, 5)), 
  trees(range = c(60, 140)),
  min_n(range = c(50, 100))
)
hpgrid <- grid_regular(hpset, levels = c(3, 2, 2))
hpgrid


# 交叉验证网格搜索过程
set.seed(32)
tune_rf <- wk_rf %>%
  tune_grid(resamples = folds,
            grid = hpgrid,
            control = control_grid(save_pred = T, verbose = F))
            
# 图示交叉验证结果
autoplot(tune_rf)
tune_rf %>%
  collect_metrics()

# 经过交叉验证得到的最优超参数
hpbest <- tune_rf %>%
  select_best(metric = "accuracy")
hpbest

# 采用最优超参数组合训练最终模型
set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest) %>%
  fit(traindata2)
final_rf

# 提取最终的算法模型
final_rf2 <- final_rf %>%
  extract_fit_engine()
plot(final_rf2, main = "随机森林树的棵树与误差演变")
legend("top", 
       legend = colnames(final_rf2$err.rate),
       lty = 1:3,
       col = 1:3,
       horiz = T)

# 变量重要性
importance(final_rf2)
varImpPlot(final_rf2, main = "变量重要性")
# 偏依赖图
partialPlot(final_rf2, 
            pred.data = as.data.frame(traindata2), 
            x.var = Age)

final_rf$fit
# 应用模型-预测训练集
predtrain_rf <- final_rf %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
predtrain_rf

# 评估模型ROC曲线-训练集上
trainroc2 <- predtrain_rf %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
autoplot(trainroc2)

# 约登法则对应的p值
yueden_rf <- trainroc2 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_rf
# 预测概率+约登法则=预测分类
predtrain_rf2 <- predtrain_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rf, "No", "Yes")))
predtrain_rf2
# 混淆矩阵
cmtrain_rf <- predtrain_rf2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_rf
autoplot(cmtrain_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtrain_rf %>%
  summary() %>%
  bind_rows(predtrain_rf %>%
              roc_auc(Lung_mets, .pred_No))



# 应用模型-预测测试集
predtest_rf <- final_rf %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_rf

# 评估模型ROC曲线-测试集上
testroc2 <- predtest_rf %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
save(testroc2,file = "testroc2.Rdata")
autoplot(testroc2)


# 预测概率+约登法则=预测分类
predtest_rf2 <- predtest_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rf, "No", "Yes")))
predtest_rf2
# 混淆矩阵
cmtest_rf <- predtest_rf2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_rf
autoplot(cmtest_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_rf %>%
  summary() %>%
  bind_rows(predtest_rf %>%
              roc_auc(Lung_mets, .pred_No))


# 合并训练集和测试集上ROC曲线
roctrain_rf %>%
  bind_rows(roctest_rf) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()
  








# 应用模型-预测测试集
predtest_rf <- final_rf %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_rf

# 评估模型ROC曲线-测试集上
Extestroc2 <- predtest_rf %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(Extestroc2)

# 预测概率+约登法则=预测分类
predtest_rf2 <- predtest_rf %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_rf, "No", "Yes")))
predtest_rf2
# 混淆矩阵
cmtest_rf <- predtest_rf2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_rf
autoplot(cmtest_rf, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_rf %>%
  summary() %>%
  bind_rows(predtest_rf %>%
              roc_auc(Lung_mets, .pred_No))
