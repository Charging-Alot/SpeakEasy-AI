import os
import sys
from execjs import get

runtime = get('Node')
js_parse = runtime.compile('''
    module.paths.push('%s');
    var parseModule = require('parse.js');
    function parseText(sentence){
        return parseModule(sentence);
    }
''' % os.path.dirname(__file__))
