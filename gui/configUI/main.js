const { app, BrowserWindow, Menu } = require('electron')



function createWindow () {   
  Menu.setApplicationMenu(null)
  // 创建浏览器窗口
  let win = new BrowserWindow({
    //开发环境与运行环境的当前文件位置不一致
    icon: "../ConfigUI/drawable/icon.png",
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  });

  // 加载index.html文件
  win.loadFile('index.html')
  // win.webContents.openDevTools({mode:'right'});
}


app.on('ready', createWindow)