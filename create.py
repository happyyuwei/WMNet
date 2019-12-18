import os
import sys
import getopt
import config
import shutil
import platform


def create_train_bootstrap(watermark, extern_script=None):
    """
    create train bootstrap
    :param watermark
    :return:
    """

    run_cmd = "python ../../train.py --env=./ --config=config.json"

    if extern_script != None:
        run_cmd = run_cmd+" --script={}".format(extern_script)

    # 检查当前系统，如果是winsows系统，则生成bat; 如果是linux系统，则生成sh
    system = platform.system()
    if system == "Windows":
        cmd = [run_cmd+"\r\n", "@pause\r\n"]
        with open("train.bat", "w") as f:
            f.writelines(cmd)
    elif system == "Linux":
        cmd = [run_cmd+"\n"]
        with open("train.bat", "w") as f:
                f.writelines(cmd)
    else:
        print("Error: System: {} is not supported currently.".format(system))
        sys.exit()


def create_config_bootstrap(app_name):
    """
    用于生成配置界面启动脚本, 配置界面功能已启用，不需要盯着晦涩的配置文件修改了。
    @since 2019.12.3
    @author yuwei
    """

    # 此命令用于隐藏cmd的黑框
    invisible_cmd = ['@echo off\r\n',
                     'if "%1" == "h" goto begin\r\n',
                     'mshta vbscript:createobject("wscript.shell").run("%~nx0 h",0)(window.close)&&exit\r\n',
                     ':begin\r\n']

    # 运行cmd
    run_cmd = "python ../../gui/ConfigUI/bootstrap.py -a {}\r\n".format(
        app_name)

    with open("config.bat", "w") as f:
        f.writelines(invisible_cmd)
        f.writelines([run_cmd])


def create_app(app_name, watermark, dataset, extern_script, conf):
    """
    create an empty app
    :param app_name:
    :param watermark
    :param dataset
    :param extern_script
    :param conf
    :return:
    """
    # create dictionary
    app_dir = os.path.join("./app/", app_name)

    # check is existed
    if os.path.exists(app_dir):
        return "error app name, the app = {} is already existed".format(app_name)

    os.mkdir(app_dir)

    # switch to app dir
    os.chdir(app_dir)

    # config_file="config.json"
    # create config file temple
    if ".json" in conf:
        config.create_config_JSON_temple(conf)
    else:
        # 目前依旧支持生成旧版配置文件， 但不推荐
        config.create_config_temple("config.txt")

    # create bootstrap train.bat
    create_train_bootstrap(watermark, extern_script)
    # create config.bat
    create_config_bootstrap(app_name)
    # create paint script paint_loss.bat
    with open("paint_loss.bat", "w") as f:
        f.writelines(["python ../../paint_loss.py"])
    # copy watermark
    if watermark != None:
        wm_path = os.path.join("../../watermark", watermark)
        if os.path.exists(wm_path):
            shutil.copyfile(wm_path, "./watermark.png")
        else:
            print("No such watermark found. Warning: watermark={}".format(watermark))

    # copy the dataset
    if dataset != None:
        dataset_list = os.listdir("../../data")
        if dataset in dataset_list:
            shutil.copytree(os.path.join("../../data", dataset), "./data")
        else:
            print("No such dataset found. Warning: dataset={}".format(dataset))
            dataset = None

    return "create app successfully, name={}, dataset={}, watermark={}, script={}".format(app_name, dataset, watermark, extern_script)


if __name__ == "__main__":
    # parse input arguments if exist

    app_name = None
    watermark = None
    dataset = None
    extern_script = None
    # 默认推荐使用JSON格式
    conf = "config.json"

    if len(sys.argv) >= 2:
        options, _ = getopt.getopt(sys.argv[1:], "hn:w:d:s:c:", [
                                   "help", "name=", "watermark=", "data=", "script=", "config="])
        print(options)
        for key, value in options:
            if key in ("-n", "--name"):
                app_name = value
            if key in ("-w", "--watermark"):
                watermark = value
            if key in ("-d", "--data"):
                dataset = value
            if key in ("-s", "--script"):
                extern_script = value
            if key in ("-c", "--config"):
                conf = value

    if app_name is not None:
        msg = create_app(app_name, watermark, dataset, extern_script, conf)
        print(msg)
    else:
        print("no name is selected, please enter you name by --name= or -n ")
