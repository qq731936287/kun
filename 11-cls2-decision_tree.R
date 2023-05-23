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
model_dt <- decision_tree(
  mode = "classification",
  engine = "rpart",
  tree_depth = tune(),
  min_n = tune(),
  cost_complexity = tune()
) %>%
  set_args(model=TRUE)
model_dt

# workflow
wk_dt <- 
  workflow() %>%
  add_model(model_dt) %>%
  add_formula(Lung_mets ~ .)
wk_dt

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 10)
folds

# 超参数寻优范围
hpset <- parameters(tree_depth(range = c(3, 7)),
                    min_n(range = c(5, 10)),
                    cost_complexity(range = c(-6, -1)))
set.seed(42)
# hpgrid <- grid_regular(hpset, levels = c(3, 2, 4))
hpgrid <- grid_random(hpset, size = 10)
hpgrid
log10(hpgrid$cost_complexity)


# 交叉验证网格搜索过程
set.seed(42)
tune_dt <- wk_dt %>%
  tune_grid(resamples = folds,
            grid = hpgrid,
            metrics=metric_set(roc_auc,pr_auc,accuracy),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_dt)
tune_dt %>%
  collect_metrics()

# 经过交叉验证得到的最优超参数
hpbest <- tune_dt %>%
  select_by_one_std_err(metric = "roc_auc", desc(cost_complexity))
hpbest

# 采用最优超参数组合训练最终模型
final_dt <- wk_dt %>%
  finalize_workflow(hpbest) %>%
  fit(traindata2)
final_dt

# 提取最终的算法模型
library(rpart.plot)
final_dt2 <- final_dt %>%
  extract_fit_engine()
rpart.plot(final_dt2)
final_dt2$variable.importance
barplot(final_dt2$variable.importance, las = 2)

# 应用模型-预测训练集
predtrain_dt <- final_dt %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
predtrain_dt

# 评估模型ROC曲线-训练集上
trainroc5 <- predtrain_dt %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
trainroc5
autoplot(trainroc5)

# 约登法则对应的p值
yueden_dt <- trainroc5 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_dt
# 预测概率+约登法则=预测分类
predtrain_dt2 <- predtrain_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_dt, "No", "Yes")))
predtrain_dt2

# 混淆矩阵
cmtrain_dt <- predtrain_dt2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_dt
autoplot(cmtrain_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
cmtrain_dt %>%
  summary() %>%
  bind_rows(predtrain_dt %>%
              roc_auc(Lung_mets, .pred_No))

# 应用模型-预测测试集
predtest_dt <- final_dt %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_dt
# 评估模型ROC曲线-测试集上
testroc5 <- predtest_dt %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(testroc5)


# 预测概率+约登法则=预测分类
predtest_dt2 <- predtest_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_dt, "No", "Yes")))
predtest_dt2

# 混淆矩阵
cmtest_dt <- predtest_dt2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_dt
autoplot(cmtest_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_dt %>%
  summary() %>%
  bind_rows(predtest_dt %>%
              roc_auc(Lung_mets, .pred_No))


# 合并训练集和测试集上ROC曲线
roctrain_dt %>%
  bind_rows(roctest_dt) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()



# 应用模型-预测测试集
predtest_dt <- final_dt %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_dt
# 评估模型ROC曲线-测试集上
Extestroc5 <- predtest_dt %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(Extestroc5)


# 预测概率+约登法则=预测分类
predtest_dt2 <- predtest_dt %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_dt, "No", "Yes")))
predtest_dt2

# 混淆矩阵
cmtest_dt <- predtest_dt2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_dt
autoplot(cmtest_dt, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
cmtest_dt %>%
  summary() %>%
  bind_rows(predtest_dt %>%
              roc_auc(Lung_mets, .pred_No))







# 重抽样验证


set.seed(42)
fitrs_dt <- 
  wk_dt %>%
  fit_resamples(folds,
                control = control_resamples(save_pred = T))
fitrs_dt

# 重抽样结果
collect_metrics(tune_dt)
collect_predictions(tune_dt) %>%
  group_by(id) %>%
  roc_curve(N_stage, .pred_N0:.pred_N1) %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(size = 1) +
  facet_wrap(~.level) +
  theme_bw()







