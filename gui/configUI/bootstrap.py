"""
该文件为配置文件界面启动项，该脚本会自动寻找特定应用目录下的config.json配置文件
由于运行时环境，配置界面代码，与配置文件存放在三个不同的文件夹中，
因此需要额外启动脚本来启动。
@author yuwei
@since 2019.12.3
"""

import os
import sys
import getopt


#该脚本有一个bat脚本调用，bat脚本存放在应用目录下，
if __name__ == "__main__":

    app="default_app"

    # 由于该启动脚本存放于配置界面代码目录下，因此每次启动需要告知配置文件所在的应用
    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "a:", ["app="])
        print(options)
        for key, value in options:
            if key in ("-a", "--app"):
                app = value

    # change dir
    os.chdir("../../gui/runtime/")

    #创建启动命令
    #配置文件窗口在ConfigUI中
    #此处相对路径为electron的运行时根目录
    boot_cmd="electron ../../gui/ConfigUI/ --config=../../app/{}/config.json".format(app)
    print(boot_cmd)
    #运行命令
    os.system(boot_cmd)

