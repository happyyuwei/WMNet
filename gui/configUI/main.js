const { app, BrowserWindow, Menu } = require('electron')



function createWindow () {   
  Menu.setApplicationMenu(null)
  // 创建浏览器窗口
  let win = new BrowserWindow({
    icon: "./drawable/icon.png",
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