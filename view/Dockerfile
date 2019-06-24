# 安装Ubuntu
FROM ubuntu
# 更新源
RUN apt-get update
# 安装Ubuntu包
RUN apt-get install -y apt-transport-https vim iproute2 net-tools ca-certificates curl wget software-properties-common
RUN apt-get update
# 安装python和pip
RUN apt-get install -y python3.6
RUN apt-get install -y python3-pip
# 工作路径
WORKDIR /ring_dsp
# 将当前目录. 生成的image复制到/app路径
COPY . /ring_dsp
# 安装运行服务需要的包
RUN pip3 install -r requirements.txt
# image的Ubuntu系统对外开放的端口
EXPOSE 80
# 终端可显示中文
ENV PYTHONIOENCODING=utf-8
# 启动container时运行服务
CMD ["python3.6", "run.py"]
