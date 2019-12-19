#coding:utf-8
import os
import time
#保存初始棋盘
shudu9X9 = []
#从文件中读取数据


def readFromFile(fileName):
    if not os.path.exists(fileName):#如果文件不存在，则新增一个空文件
        return -1
    with open(fileName,"r",encoding="utf-8") as f:
        for one in f:
            oneLine = one.strip("\n").split(",")
            for i in range(len(oneLine)):#将所有数据转换为数值型
                oneLine[i] = int(oneLine[i])
            shudu9X9.append(oneLine)
    return 1


#打印棋盘
def printMe(map):
    for i in range(9):
        print("   %-1d"%(i+1),end="")#打印列号
    print()
    print(" ┌─────────────────┐")
    for i in range(9):
        print("\033[0m%d"%(i+1),end="")#打印行号
        for j in range(9):
            if j ==3 or j ==6 :#打印粗线，用于分割3X的格子
                print("\033[0;34m┃",end="")
            else:
                print("\033[0m│",end="")

            if map[i][j] == 0:
                print(" \033[0m ", end="")
            else:
                print(" \033[0;31m%d" %map[i][j], end="")#红色
        print("\033[0m│")
        if i!= 8:
            if i==2 or i==5:#打印粗线，用于分割3X3的格子
                print(" \033[0;34m┣━━━━━━━━━━━━━━━━━┫")
            else:
                    print(" \033[0m├─────────────────┤")

    print(" └─────────────────┘")


#执行游戏
def executeGame(diff):
    #将棋盘的内容拷贝一份
    exeGameList = shudu9X9.copy()

    while True:
        flag = 0
        #打印棋盘
        printMe(exeGameList)
        try:
            row = int(input("请输入行号（1~9）："))
            column = int(input("请输入列号（1~9）："))
            iNum = int(input("请输入数字（1~9）："))
        except:
            print("输入的行号、列号、数字等有误，请重新输入。")
            continue
        # 输入的行列号超出范围
        if row>9 or row<1 or column>9 or column<1 or iNum > 9 or iNum < 1:
            print("输入的行或列号或值超出范围，请重新输入")
            continue
        #判断该位置是否为空
        if exeGameList[row-1][column-1] != 0:
            print("该位置已有子，请重新输入。")
            continue
        #判断同一行是否有重复
        for j in range(9):
            if exeGameList[row-1][j-1] == iNum:
                print("同行内重复，出子错误，请重新输入。")
                flag = -1#错误标记
                break
        # 判断同一列是否有重复
        if flag!= -1:
            for i in range(9):
                if exeGameList[i][column-1] == iNum:
                    print("同列内重复，出子错误，请重新输入。")
                    flag = -1  # 错误标记
                    break
        #判断自己的9宫格中是否有重复
        if flag != -1:
            indexRow = (row-1)//3
            indexColumn = (column-1)//3
            for i in range(indexRow*3,indexRow*3+3):
                for j in range(indexColumn*3,indexColumn*3+3):
                    if exeGameList[i][j] == iNum:
                        print("9宫格内数字重复，请重新输入。")
                        flag = -1# 错误标记
                        break
        #判断是否结束
        if flag!= -1:
            exeGameList[row-1][column-1] = iNum
            #判断是否结束
            for i in range(9):
                if 0 in exeGameList[i]:
                    break
            else:
                print("恭喜，获胜！！")
                break


if __name__ == "__main__":
    while True:
        print("**************************")
        print("***欢迎来到数独游戏********")
        print("*1.简单*******************")
        print("*2.中等*******************")
        print("*3.困难*******************")
        print("**************************")
        select = int(input("请选择游戏难度："))
        if select == 1 or select == 2 or select == 3:#难易度
            #根据不同选择，加载不同难度棋盘
            if -1 == readFromFile("easy.txt" if select==1 else "normal.txt"  if select==2 else "hard.txt"):
                print("棋盘加载错误，请联系作者。")
                exit()
            else:
                print("棋盘加载中......")
            time.sleep(2)#暂停两秒
            #执行数独功能
            executeGame(select)
        else:
            print("输入错误，请重新输入。")
            continue