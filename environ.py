# -*- coding: utf-8 -*-
'''
Global environment for the entire application: global objects, settings (combined config_global & config_local) etc.
Global initialization.

@author:  Marcin Wojnarski
@contact: mwojnars@ns.onet.pl 
'''

from nifty.util import class2dict

from fireweb.db.database import MySQL
from fireweb.db.model import Model
from fireweb.db.filebase import FileBase, FileModel

import config

# global Database object, for use in all queries around the entire application; supports multi-threading
print "environ.py: connecting to DB..."
db = Model._db = MySQL(threaded = True, **class2dict(config.db))

# global FileBase, which extends relational 'db' onto filesystem-based data storage
filebase = FileModel._base = FileBase(config.filebase)

