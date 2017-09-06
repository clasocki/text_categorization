import os
from paperity.environ import db
import csv
from os.path import exists
import logging
from collections import defaultdict
import random
from langdetect import detect_langs
from semantic_model import DocumentIterator

from langdetect import DetectorFactory
DetectorFactory.seed = 0

logging.basicConfig(filename='retrieve_save.log')
logger = logging.getLogger('retrieve_save_logger')

def choose_english_docs_only():
    batch_size = 50000
    document_iterator = DocumentIterator(document_batch_size=None, db_window_size=batch_size, doc_filter="1 = 1")
    not_en_cnt, docs_cnt = 0, 0 
    for doc in document_iterator.getAllInBatches():
        docs_cnt += batch_size
        print docs_cnt
        try:
            if doc.rawtext is not None and len(doc.rawtext) > 0:
                detected_langs = detect_langs(doc.rawtext)
                print doc.id, detected_langs
                """"
                if not ('en' in [l.lang for l in detected_langs] and len(detected_langs) == 1):
                    sql = "update pap_papers_1 set published = 0 where id = %s" % (doc.id, )
                    print sql
                    db.query(sql)
                    db.commit()

                    not_en_cnt += 1
                    print not_en_cnt
                """
        except:
            pass

def retrieve_rawtexts_by_id(srcfilename, destdir):
    """
    Extract 'rawtext' for all papers which ids are provided in the file srcfilename, each id kept in a separate line,
    and save them to separate files in the provided directory: <destdir>/<pap_id>.txt
    """

    if not exists(destdir):
        raise Exception("No such directory: '" + destdir + "'")        

    paper_ids = []

    f_ids = open(srcfilename, 'r')
    for paper_id in f_ids:
        paper_ids.append(paper_id)

    f_ids.close()

    papers = Paper.select("pid, rawtext", _where = "pid in (%s)" % ', '.join(paper_ids))

    path = destdir + '/%s.txt'
    for paper in papers:
        f_name = path % paper.pid
        f = open(f_name, 'w')
        
        print >>f, paper.rawtext.encode('utf-8')

        f.close()

def compare_category_allocation(filename1, filename2, filename_same, filename_diff):
    allocations1, allocations2 = dict(), dict()
    
    num_invalid_titles = 0
    with open(filename1, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', lineterminator='\n')

        for row in reader:
            if len(row) == 4:
                pap_id = row[0]
                title = row[2]
                category = row[3]
                allocations1[title] = set(category.split(','))
            else:
                num_invalid_titles += 1

    with open(filename2, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', lineterminator='\n')        
        for row in reader:
            if len(row) == 4:
                pap_id = row[0]
                title = row[2]
                category = row[3]
                allocations2[title] = set(category.split(','))
            else:
                num_invalid_titles += 1

    matches = 0
    papers_in_both_sets = 0
    print len(allocations1), len(allocations2)
    with open(filename_same, 'w') as f_same, open(filename_diff, 'w') as f_diff:
        same_csv_writer = csv.writer(f_same)
        diff_csv_writer = csv.writer(f_diff)
        for title, category in allocations2.iteritems():
            #print title, category
            if title in allocations1:
                papers_in_both_sets += 1
                if allocations1[title].intersection(category):
                    matches += 1
                    same_csv_writer.writerow([title, category, allocations1[title]])
                else:
                    diff_csv_writer.writerow([title, category, allocations1[title]])

    print "Matching categories: %s out of %s, which accounts for %s" % (matches, papers_in_both_sets, float(matches) / papers_in_both_sets)
    print "Invalid: %s" % (num_invalid_titles, )

def save_rawtexts_with_categories(rawtext_dirname, paper_list_filename):
    if not exists(rawtext_dirname):
        raise Exception("No such directory: '" + rawtext_dirname + "'")

    if not exists(paper_list_filename):
        raise Exception("No such file: '" + paper_list_filename + "'")       

    with open(paper_list_filename, 'r') as paperfile:
        reader = csv.reader(paperfile, delimiter=',', quotechar='"')
        reader.next() #skipping header
        
        for i, paper_row in enumerate(reader):
            print i
            paper_id = paper_row[4]
            #paper_title = paper_row[2].replace("'", "''").decode('utf-8')
            paper_title = ''
            #paper_category = paper_row[0] + "," + paper_row[1]
            paper_category = paper_row[1]
            rawtext_filename = '%s/%s.txt' % (rawtext_dirname, paper_id)
            
            if exists(rawtext_filename):
                rawtext_file = open(rawtext_filename, 'r')
                paper_rawtext = rawtext_file.read().replace("'", "''").decode('utf-8')
                rawtext_file.close()
                db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE pid = " + str(paper_id))[0]['COUNT(1)']
                
                sql1 = ""
                sql2 = ""
                sql3 = ""

                if db_row_count > 0:
                    #sql1 = "update pap_papers_1 set published = 0 where pid = %s" % (paper_id, )
                    sql2 = "update pap_papers_2 p2 left outer join pap_papers_1 p1 on p1.id = p2.id "
                    sql2 += "set p2.title = '%s', p2.learned_category = '%s' where p1.pid = %s" % (paper_title, paper_category, paper_id)
                    #sql3 = "update pap_papers_3 p3 left outer join pap_papers_1 p1 on p1.id = p3.id "
                    #sql3 += "set p3.rawtext = '%s' where p1.pid = %s" % (paper_rawtext, paper_id)
                else:
                    #new_id = int(db.select("select max(id) from pap_papers_1")[0]['max(id)']) + 1
                    #sql1 = "insert into pap_papers_1(id, pid,journal_id,published) values(%s, %s, 1001, 0)" % (new_id, paper_id,)
                    #sql2 = "insert into pap_papers_2(id, title,learned_category) values(%s, '%s', '%s')" % (new_id, paper_title, paper_category)
                    #sql3 = "insert into pap_papers_3(id, rawtext) values(%s, '%s')" % (new_id, paper_rawtext)
                    pass
                print sql2
                
                #db.query(sql1)
                db.query(sql2)
                #db.query(sql3)
    
    db.commit()

def save_or_update_rawtexts(rawtext_dirname):
    if not exists(rawtext_dirname):
        raise Exception("No such directory: '" + rawtext_dirname + "'")
    
    errors = []
    for subdir, dirs, files in os.walk(rawtext_dirname):
        for f in files:
            success = False
            with open(os.path.join(rawtext_dirname, f), 'r') as f_obj:
		paper_rawtext = f_obj.read().replace("'", "''").decode('utf-8')
                paper_title = f
                paper_id = f.split('.')[0]
                paper_ext = f.split('.')[1]
                if 'txt' not in f: continue

                db_row_count = db.select("SELECT COUNT(1) FROM pap_papers_view WHERE pid = " + str(paper_id))[0]['COUNT(1)']
                
                sql1 = ""
                sql2 = ""
                sql3 = ""

                if db_row_count > 0:
                    pass
                    #sql1 = "update pap_papers_1 set published = 1 where pid = %s" % (paper_id, )
                    #sql2 = "update pap_papers_2 p2 left outer join pap_papers_1 p1 on p1.id = p2.id "
                    #sql2 += "set p2.title = '%s', p2.learned_category = '%s' where p1.pid = %s" % (paper_title, paper_category, paper_id)
                    #sql3 = "update pap_papers_3 p3 left outer join pap_papers_1 p1 on p1.id = p3.id "
                    #sql3 += "set p3.rawtext = '%s' where p1.pid = %s" % (paper_rawtext, paper_id)
                else:
                    new_id = int(db.select("select ifnull(max(id), 0) max from pap_papers_1")[0]['max']) + 1
                    sql1 = "insert into pap_papers_1(id, pid,journal_id,published) values(%s, %s, 1001, 1)" % (new_id, paper_id,)
                    sql2 = "insert into pap_papers_2(id, title,learned_category) values(%s, '%s', '%s')" % (new_id, paper_title, '')
                    sql3 = "insert into pap_papers_3(id, rawtext) values(%s, '%s')" % (new_id, paper_rawtext)
                try:
                    if sql1:
                        db.query(sql1)
                        db.query(sql2)
                        db.query(sql3)
                    
                        db.commit()
                    success = True
                except Exception, ex:
                    logger.error('[Error]: ' + str(ex))
                    pass
            if success:
                pass
                #os.remove(os.path.join(rawtext_dirname, f))

def save_rawtexts(rawtext_dirname):
    if not exists(rawtext_dirname):
        raise Exception("No such directory: '" + rawtext_dirname + "'")
    
    errors = []
    existing_pids = set(elem['pid'] for elem in db.select("SELECT pid FROM pap_papers_view"))
    for subdir, dirs, files in os.walk(rawtext_dirname):
        for f in files:
            success = False
            with open(os.path.join(rawtext_dirname, f), 'r') as f_obj:
                paper_rawtext = f_obj.read().replace("'", "''").decode('utf-8')
                paper_title = f
                paper_id = f.split('.')[0]
                paper_ext = f.split('.')[1]
                if 'txt' not in f: continue

                sql1 = ""
                sql2 = ""
                sql3 = ""

                if int(paper_id) not in existing_pids:
                    new_id = int(db.select("select ifnull(max(id), 0) max from pap_papers_1")[0]['max']) + 1
                    sql1 = "insert into pap_papers_1(id, pid,journal_id,published) values(%s, %s, 1001, 1)" % (new_id, paper_id,)
                    sql2 = "insert into pap_papers_2(id, title,learned_category) values(%s, '%s', '%s')" % (new_id, paper_title, '')
                    sql3 = "insert into pap_papers_3(id, rawtext) values(%s, '%s')" % (new_id, paper_rawtext)
                try:
                    if sql1:
                        db.query(sql1)
                        db.query(sql2)
                        db.query(sql3)
                         
                        print "Added: " + str(paper_id)
                        db.commit()
                    success = True
                except Exception, ex:
                    logger.error('[Error]: ' + str(ex))
                    pass
            if success:
                os.remove(os.path.join(rawtext_dirname, f))

def extract_validation_set(paper_list_filename):
    with open(paper_list_filename, 'r') as paperfile:
        reader = csv.reader(paperfile, delimiter=',', quotechar='"')
        reader.next() #skipping header

        category_pid_map = defaultdict(list)
        
        for i, paper_row in enumerate(reader):
            paper_id = paper_row[4]
            paper_subcategory = paper_row[2] 
            category_pid_map[paper_subcategory].append((paper_id, paper_row))

        validation_set, training_set = [], []
        for category, ids in category_pid_map.iteritems():
            validation_sample = random.sample(ids, len(ids) / 2)
            validation_ids, validation_rows = set(), []
            
            for id, row in validation_sample:
                validation_ids.add(id)
                validation_rows.append(row)

            for id, row in ids:
                if id not in validation_ids:
                    training_set.append(row)

            validation_set.extend(validation_rows)

        with open('training_set.csv', 'w') as trainfile, open('validation_set.csv', 'w') as validfile:
            train_writer = csv.writer(trainfile)
            valid_writer = csv.writer(validfile)

            for row in training_set:
               train_writer.writerow(row)

            for row in validation_set:
               valid_writer.writerow(row)

if __name__ == "__main__":
    rawtext_dirname = "/home/clasocki/paperity_full_rawtexts"
    paper_list_filename = "/home/clasocki/training_set.csv"

    choose_english_docs_only()
    #save_rawtexts_with_categories(rawtext_dirname, paper_list_filename)
    #save_rawtexts(rawtext_dirname)
    #extract_validation_set(paper_list_filename)

    #dirname = '/home/clasocki/'
    #compare_category_allocation(dirname + 'labeled_iterative.csv', dirname + 'labeled_iter2.csv', dirname + 'same_categories.csv', dirname + 'diff_categories.csv')
