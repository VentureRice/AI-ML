#—UTF-8—
###############################载入需要的包#############################################################
library(stringr)
library(r2pmml)
library(plyr)
library(dplyr)
library(devtools)
library(lubridate)
library(data.table)
library(openxlsx)
Sys.setlocale("LC_ALL", "zh_cn.utf-8")

###############################载入需要的数据集############################################################
# 载入告警数据
alarm = read.xlsx("~/Desktop/ele/长期数据/告警.xlsx",sheet = 2,colNames = T)
# 载入静态指标
index = read.csv("~/Desktop/ele/长期数据/静态指标字段.csv",header = F)
# 命名列名
colnames(index) = c('id','liftName','brandName','limitSpeed','limitWeight','makeDate','placetypename')
colnames(alarm) = c("id","start","end")
alarm = alarm%>%as.data.table()
###############################数据清洗#################################################################
# 转化数据格式
alarm[,Date := as.Date(start,tz="Asia/shanghai")]
alarm[,start := as.POSIXct(start)]
alarm[,end := as.POSIXct(end)]
# 读取电梯数据的list
status_list = list.files("~/Desktop/ele/长期数据/status")
col_status = c("id","collecttime","s_speed")
# 整理得到结构化数据
status_res = NULL
for (i in 1:length(status_list)) {
  # 读入data.table格式的数据
  status = fread(paste0("~/Desktop/ele/长期数据/status/",status_list[i]),sep=",")
  colnames(status) = col_status
  # 转化时间格式
  status[,collecttime := as.POSIXct(round(collecttime/1000),origin = "1970-01-01")]
  # 记录电梯id-elevatorID
  elevatorID = status$id[1]
  # 将status按照时间升序
  setorder(status,collecttime)
  # 去除掉重复的记录
  status = unique(status)
  # 计算加速度-向量作差-dv
  status$dv = 0
  status$dv[-1] = (status$s_speed[-1]-status$s_speed[-length(status$s_speed)])/as.numeric(difftime(status$collecttime[-1],status$collecttime[-length(status$collecttime)],units = "sec"))
  # 删除掉缺失值
  status= status[(dv!=-Inf & !is.na(dv)),]
  # 滑动窗口
  Timeline = seq(from = max(status$collecttime)-60*60*24*30, to = max(status$collecttime)-60*60*24*7,by = 60*60*24*1)
  for (j in 1:length(Timeline)) {#
    Time = Timeline[j]
    # 去掉缺失较多的时间段
    if (as.POSIXct(Time) %in% status$collecttime | as.POSIXct(Time-100) %in% status$collecttime|| as.POSIXct(Time-1) %in% status$collecttime){
      # 筛选某时间点之前的status
      status0 = status[collecttime<Time,]
      # 重新排序
      setorder(status0,collecttime)
      # 记录小时
      status0[,hour := hour(collecttime)]
      # 关闭warnings
      warnings('off')
      # 整理结构化数据-data.table
      status_sum0 = status0[,.(max_speed_up = max(s_speed),# 最大向上速度
                               max_speed_down = min(s_speed),# 最大向下速度
                               max_dv_up = max(dv),# 最大向上加速度
                               max_dv_down = min(dv),# 最大向下加速度
                               sd_speed_up = sd(s_speed[s_speed>0]),# 向上速度标准差
                               sd_speed_down = sd(s_speed[s_speed<0]),# 向下速度标准差
                               sd_dv_up = sd(dv[dv>0]),# 向上加速度标准差
                               sd_dv_down = sd(dv[dv<0]),# 向下加速度标准差
                               ratio_up = length(s_speed[s_speed>0])/length(s_speed),# 向上运动占比
                               ratio_down = length(s_speed[s_speed<0])/length(s_speed),# 向下运动占比
                               ratio_speedup = length(dv[dv>0])/length(dv),# 加速占比
                               ratio_speeddown = length(dv[dv<0])/length(dv),# 减速占比
                               rush_hour1 = length(s_speed[s_speed!=0&hour>=7&hour<=9])/
                                 length(s_speed&hour>=7&hour<=9),# 早高峰运动时间占比
                               rush_hour2 = length(s_speed[s_speed!=0&hour>=15&hour<=18])/
                                 length(s_speed&hour>=15&hour<=18))]#晚高峰运动时间占比
      status_res0 = cbind(id=elevatorID,timepoint=Time,status_sum0)
      status_res = rbind(status_res,status_res0)
    }
    # 进度条
    cat('已完成：',round(i/length(status_list),5),"\r")
  }
}

# 上一周故障次数
status_res$pre_bad = 0
# 未来一周故障次数
status_res$if_bad = 0
status_res$timepoint = as.POSIXct(status_res$timepoint)
for (i in 1:nrow(status_res)) {
  # 记录电梯id
  ids = status_res$id[i]
  # 调取告警时间
  alarm_record = alarm[id == ids,]
  if (nrow(alarm_record)>0){
    for (j in 1:nrow(alarm_record)){
      # 提取因变量-if_bad
      if (status_res$timepoint[i]+7*60*60*24<alarm_record$start[j] &
          status_res$timepoint[i]+14*60*60*24>alarm_record$start[j]) {
        status_res$if_bad[i] = 1
      }
      # 提取自变量-pre_bad-本周内的故障次数
      if (status_res$timepoint[i]-7*60*60*24<alarm_record$start[j] &
          status_res$timepoint[i]>alarm_record$start[j]) {
        status_res$pre_bad[i] = status_res$pre_bad[i] + 1
      }
    }
  }
}


write.csv(status_res,"~/Desktop/ele/status_res.csv")
