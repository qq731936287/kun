# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/logistic_reg.html
# https://parsnip.tidymodels.org/reference/details_logistic_reg_glm.html

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

traindata %>%
  mutate(across(1:9,~rank(.x))) %>%
  cor(method = "spearman") %>%
  corrplot::corrplot(col=colorRampPalette(c("cornsilk","white","firebrick4"))(200),method="color",order="AOE",addCoef.col="black",tl.col = "black",tl.cex=1,number.font=0.6,number.cex=0.6
  )


datarecipe <- recipe(Lung_mets ~ ., traindata) %>%
  step_naomit(all_predictors(), skip = F) %>%
  prep()
datarecipe


# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL)
testdata2 <- bake(datarecipe, new_data = testdata)
test2 <- bake(datarecipe, new_data = test)

save(traindata2,file = "traindata2.Rdata")
save(testdata2,file = "testdata2.Rdata")

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

# 训练模型
# 设定模型
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic

# 拟合模型
fit_logistic <- model_logistic %>%
  fit(Lung_mets ~ ., traindata2)
fit_logistic
fit_logistic$fit
summary(fit_logistic$fit)

# 系数输出
fit_logistic %>%
  tidy()

# 应用模型-预测训练集
predtrain_logistic <- fit_logistic %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(traindata2 %>% select(Lung_mets))
predtrain_logistic


# 评估模型ROC曲线-训练集上
trainroc1 <- predtrain_logistic %>%
  roc_curve(Lung_mets, .pred_No, event_level = "first") %>%
  mutate(dataset = "train")
trainroc1
autoplot(trainroc1)

# 约登法则对应的p值
yueden_logistic <- trainroc1 %>%
  mutate(yueden = sensitivity + specificity - 1) %>%
  slice_max(yueden) %>%
  pull(.threshold)
yueden_logistic
# 预测概率+约登法则=预测分类
predtrain_logistic2 <- predtrain_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_logistic, "No", "Yes")))
predtrain_logistic2

# 混淆矩阵
cmtrain_logistic <- predtrain_logistic2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtrain_logistic
autoplot(cmtrain_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
cmtrain_logistic %>%
  summary() %>%
  bind_rows(predtrain_logistic %>%
              roc_auc(Lung_mets, .pred_No))

# 应用模型-预测测试集
predtest_logistic <- fit_logistic %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_logistic

# 评估模型ROC曲线-测试集上
testroc1 <- predtest_logistic %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(testroc1)

# 预测概率+约登法则=预测分类
predtest_logistic2 <- predtest_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_logistic, "No", "Yes")))
predtest_logistic2

# 混淆矩阵
cmtest_logistic <- predtest_logistic2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_logistic

autoplot(cmtest_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
cmtest_logistic %>%
  summary() %>%
  bind_rows(predtest_logistic %>%
              roc_auc(Lung_mets, .pred_No))


# 合并训练集和测试集上ROC曲线
trainroc2 %>%
  bind_rows(testroc2) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(size = 1) +
  theme_bw()


# 应用模型-预测测试集
predtest_logistic <- fit_logistic %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(testdata2 %>% select(Lung_mets))
predtest_logistic

# 应用模型-预测测试集
predtest_logistic <- fit_logistic %>%
  predict(new_data = test2, type = "prob") %>%
  bind_cols(test2 %>% select(Lung_mets))
predtest_logistic
# 评估模型ROC曲线-测试集上
Extestroc1 <- predtest_logistic %>%
  roc_curve(Lung_mets, .pred_No) %>%
  mutate(dataset = "test")
autoplot(Extestroc1)


# 预测概率+约登法则=预测分类
predtest_logistic2 <- predtest_logistic %>%
  mutate(.pred_class = 
           factor(ifelse(.pred_No >= yueden_logistic, "No", "Yes")))
predtest_logistic2
# 混淆矩阵
cmtest_logistic <- predtest_logistic2 %>%
  conf_mat(truth = Lung_mets, estimate = .pred_class)
cmtest_logistic

autoplot(cmtest_logistic, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))

# 合并指标
cmtest_logistic %>%
  summary() %>%
  bind_rows(predtest_logistic %>%
              roc_auc(Lung_mets, .pred_No))





