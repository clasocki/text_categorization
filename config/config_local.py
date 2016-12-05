# -*- coding: utf-8 -*-
'''
Local (for a given installation) changes in the configuration of the application.
Copy this file as config_local.py in the same folder and make any changes you need.

@author:  Marcin Wojnarski
@contact: mwojnars@ns.onet.pl 
'''

# KEEP THIS LINE!
from config_global import *


# NOW, MAKE ANY CHANGES YOU NEED TO THE GLOBAL CONFIG, OR ADD NEW SETTINGS ...

db.user = "root"
db.password = "1qaz@WSX"

uwsgi.processes = 2
uwsgi.threads = 2

backupFTP.host = "ftpback-rbx6-41.ovh.net"
backupFTP.user = "ns3000749.ip-37-59-46.eu"
backupFTP.password = "..."

# FTP connection in Midnight Commander:
# ftp://ns3000749.ip-37-59-46.eu:<PASS>@ftpback-rbx6-41.ovh.net
