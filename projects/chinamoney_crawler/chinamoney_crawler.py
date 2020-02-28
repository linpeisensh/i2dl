from requests_html import HTMLSession
import requests
import time
import os
import re
from PyPDF2 import PdfFileReader

def list_to_txt(list_name, file_name):
    txt = open('{}.txt'.format(file_name), 'w')
    for s in list_name:
        s = s + '\n'
        txt.write(s)
    txt.close()
    print("保存{}.txt成功".format(file_name))

def isValidPDF_pathfile(pathfile):
    bValid = True
    reader = None
    try:
        reader = PdfFileReader(pathfile)

    except:
        bValid = False
        if reader != None:
            if reader.isEncrypted:
                bValid = True

    return bValid

def companys_list(file):
    f = open(file,mode = 'r+', encoding='utf-8') #需要查询公司的名字
    line = f.readline().strip('\n')
    companys_name = []
    while line:
        companys_name.append(line)
        line = f.readline().strip('\n')
    f.close()
    return companys_name

def recheck(empty,recheck_times,i,recheck_point):
    if all(emp == 1 for emp in empty) and recheck_times >= 1:
        recheck_point = i
        i -= 2
        recheck_times -= 1
        time.sleep(60)
        print('连续2个空文件夹，重确定开始')
    if i == recheck_point and recheck_times == 0:
        recheck_times = 1
        print('重确定结束')
    return recheck_times, i, recheck_point

def pdf_name_and_url_list(j,company_name):
    url = 'http://www.chinamoney.com.cn/ses/rest/cm-u-notice-ses-cn/query?sort=date&text={}&date=all&field=title&start=&end=&pageIndex={}&pageSize=15&public=false&infoLevel=RJslflquHKcMynhGjaVhuXdBJhuP2PkcxxG5nMJCFFlTqul0Zn2tzRGvT9DmoxY0kvq7XGb5hZqb%0A5S/yVluzAR9r9GALRXVWlFTSDw4ESWjR6KqdAvMbDXxygumjCcBeNs5sbj8uQuk/qJIm52N/Gbn1%0AvLgPfDiFc%2BNS8dvlpho=%0A&sign=Q/d8solfMh3GOoMI5WmGUaZA1ukiCpO5sMwap9ByMZnt4tsJZeSkX6Wq1v3lRrKsnQLcWdAPun00%0ALsYa5AtcTZpCs2CvuKf8xTKL5JKkAphGIIEbpsADAhjeg2dCZIBVMUOFd2LaiLvRLJLML9AfJTc/%0AI44XV2MvFkyyEBuTLsA=%0A&channelIdStr=2496,2833,2632,2589,2663,2556,2850,2884,2900,3300,2496,2833,2632,2589,2663,2556,2850,2884,2900,3300&nodeLevel=1'.format(
        company_name, j)
    session = HTMLSession()
    r = session.get(url, headers=headers)
    session.close()

    pdf_urls_list = r.html.search_all('"dealPath":"{}","uuid"')
    pdf_names_list = r.html.search_all('"title":"{}","author"')
    return pdf_urls_list, pdf_names_list

def no_more_page(i,j, list_len,empty,company_name):
    stop = False
    if list_len == 0:
        print('{}.{}的文件已下载好'.format(i, company_name))
        stop = True
        if j == 1:
            empty[0] = empty[1]
            empty[1] = 1
        else:
            empty[0] = empty[1]
            empty[1] = 0

    return stop, empty

def pdf_name_and_url(pdf_names_list,pdf_urls_list,prior11,prior12,no_prior_report):
    pdf = {}
    for k in range(len(pdf_names_list)):
        pdf_name = pdf_names_list[k]
        if prior(prior11,pdf,no_prior_report,pdf_name) or prior(prior12,pdf,no_prior_report,pdf_name):
            url_link = pdf_urls_list[k][0]
            pdf[pdf_name] = url_link
    return pdf

def prior(p,pdf,no_prior_report,pdf_name):
    if p[0] in pdf_name and any(report in pdf_name for report in p[1:]) and pdf_name not in pdf.keys() \
            and not any(no in pdf_name for no in no_prior_report):
        return True
    else:
        return False

def raw_url(pdf_url):
    retry_times = 0
    raw_file_url = None
    while raw_file_url == None and retry_times <= 10:
        if retry_times != 0:
            print('第{}次重试'.format(retry_times))
        retry_times += 1
        time.sleep(5)
        session = HTMLSession()
        pdf_r = session.get(pdf_url, headers=headers)
        pdf_r.html.render(scrolldown=50, sleep=0.2, reload=True, timeout=8)
        session.close()
        raw_file_url = pdf_r.html.search('makeFDLT(\'{}\');')
    return raw_file_url, retry_times

def download_pdf(validPDF,file_url,file_name,folder):
    redownload_time = 1
    broken_pdf = []
    while not validPDF and redownload_time <= 5:
        print('{}. {} 下载'.format(redownload_time, file_name))
        redownload_time += 1
        r = requests.get(file_url, stream=True)
        with open(file_name, "wb") as file:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        time.sleep(3)
        file.close()
        validPDF = isValidPDF_pathfile(file_name)
    if not validPDF:
        broken_pdf.append(folder)
        broken_pdf.append(file_url)
    return broken_pdf

def crawler(root_folder,original_file,no_prior_report,prior11,prior12,stoppoint,recheck_times):
    broken_pdf = []
    no_getpdf = []
    i = stoppoint
    empty = [0,0]
    recheck_point = stoppoint
    companys_name = companys_list(original_file)
    while i < len(companys_name):
        recheck_times, i, recheck_point = recheck(empty, recheck_times, i, recheck_point)
        company_name = companys_name[i]
        i += 1
        folder = '%d.' % (i) + company_name
        if (os.path.exists(folder) == False):
            os.mkdir(folder)
        os.chdir(folder)

        for j in range(1, 520):  # 该网站15个pdf一页，j为页数
            pdf_urls_list, pdf_names_list = pdf_name_and_url_list(j, company_name)
            list_len = len(pdf_names_list)
            stop, empty = no_more_page(i, j, list_len, empty,company_name)
            if stop:
                break
            for k in range(list_len):
                x = pdf_names_list[k][0]
                pdf_names_list[k] = re.sub(r1, '', x)

            pdf = pdf_name_and_url(pdf_names_list, pdf_urls_list, prior11, prior12, no_prior_report)

            if pdf == {}:
                print('{}页{}无目标文件'.format(j, company_name))
                continue

            for pdf_name, url_link in pdf.items():
                file_name = "{}{}.pdf".format(i, pdf_name)
                validPDF = isValidPDF_pathfile(file_name)
                if os.path.exists(file_name) and validPDF:
                    continue

                pdf_url = 'http://www.chinamoney.com.cn' + url_link
                raw_file_url, retry_times = raw_url(pdf_url)

                if retry_times > 10:
                    print(pdf_url)
                    no_getpdf.append('{}.{}'.format(i + 1, company_name))
                    no_getpdf.append(pdf_name)
                    no_getpdf.append(pdf_url)
                    continue

                file_url = 'http://www.chinamoney.com.cn/dqs/cm-s-notice-query/' + raw_file_url[0].replace('amp;', '')
                broken_pdf += download_pdf(validPDF, file_url,file_name,folder)
        os.chdir(root_folder)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36"}


root_folder = 'C:\\Users\linpeisen\PycharmProjects\day2\doit\github'
os.chdir(root_folder)
original_file = "companyname.txt"

no_prior_report = ['决定','摘要','季度','半年']

prior11 = ['2018','年度报告','财务','审计']
prior12 = ['2019','评级报告']

broken_pdf = []
no_getpdf = []
stoppoint = 0

empty = [0,0] # 0 最近2个文件夹不空
recheck_times = 1



r1 = '[a-zA-Z\'/()*<=>?\\\\ "]+' # 去除文件中所有非数字和中文
crawler(root_folder,original_file,no_prior_report,prior11,prior12,stoppoint,recheck_times)



list_to_txt(no_getpdf,'需再更新部分')
list_to_txt(broken_pdf,'损坏的pdf')