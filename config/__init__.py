# -*- coding: utf-8 -*-
'''
Combined global+local settings of the application.
In modules that want to import config settings, you usually should use:

import paperity.config as config

Normally, you should NOT import directly config_global or config_local in application code.

@author:  Marcin Wojnarski
@contact: mwojnars@ns.onet.pl 
'''

from config_global import *

try:
    from config_local import *          # config_local should internally import * from config_global, but it may be missing - thus catching the exception

except: pass
