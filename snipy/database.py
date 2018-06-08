# -*- coding: utf-8 -*-
from contextlib import closing
import MySQLdb
import MySQLdb.cursors
from _mysql_exceptions import OperationalError
from .basic import patch


def db_config(db):
    """
    :param db: str: 디비 스키마
    :return: mysql connection 공통 옵션
    """
    return {
        'host': '127.0.0.1',
        'user': "default_user",
        'passwd': "default_pass",
        'charset': 'utf8',
        'db': db
    }


def database(db='', **kwargs):
    """
    usage:
    with database('my_db') as conn:
        c = conn.cursor()
        ....
    database 커넥션 with 문과 같이 사용하고, 알아서 close하기
    :param db: str: db스키마
    :param kwargs:
    :return:
    """
    db = kwargs.pop('db', db)
    arg = db_config(db)
    arg.update(kwargs)

    return closing(MySQLdb.connect(**arg))


def connect(db='', **kwargs):
    """
    db 접속 공통 인자들 채워서 접속, schema만 넣으면 됩니다.
    db connection 객체 반환이지만
    with 문과 같이 쓰이면 cursor임에 주의 (MySQLdb의 구현이 그렇습니다.)
    ex1)
    import snipy.database as db
    conn = db.connect('my_db')
    cursor = conn.cursor()

    ex2)
    import snipy.database as db
    with db.connect('my_db') as cursor:
        cursor.execute(query)

    :param db: str: db schema
    :param kwargs: 추가 접속 정보
    :return: connection or cursor
    """
    arg = db_config(db)
    arg.update(kwargs)
    return MySQLdb.connect(**arg)


def _cursor_exit(cursor, exc_type, exc_value, traceback):
    """
    cursor with문과 쓸수 있게 __exit__에 바인딩
    :param cursor:
    :param exc_type:
    :param exc_value:
    :param traceback:
    :return:
    """
    if exc_type is not None:
        print(exc_value, traceback)
    cursor.connection.close()


def cursor(db, **kwargs):
    """
    db cursor 리턴
    :param db: str: db 스키마 이름
    :param kwargs: 추가 MySQLdb.connect 의 kwarg들
    :return: cursor 오브젝트
    """
    return connect(db, **kwargs).cursor()

# 커서를 with문과 쓸수 있게 patch
setattr(MySQLdb.cursors.Cursor, '__enter__', lambda cur: cur)
setattr(MySQLdb.cursors.Cursor, '__exit__', _cursor_exit)


def execute(db, query, args=None, **kwargs):
    c = cursor(db, **kwargs)
    # logging.info(query)
    try:
        c.execute(query, args=args)
    except OperationalError as e:
        print(query, args)
        print(e.message, e.args)
        raise e
    return c


def fetch(query, args=None, **kwargs):
    """
    for record in fetch(query, args, **configs):
        print record
    :param args:
    :param db: str: db 스키마
    :param query: 쿼리 스트링
    :param kwargs: db connection 추가 인자. 보통 생략
    :return: iterator
    """
    cur = execute(kwargs.pop('db', ''), query, args, **kwargs)

    for r in cur:
        yield r

    cur.connection.close()

# """
# cursor method patching
#
# """


def fieldcount(cursor):
    return len(cursor.description)


def fieldnames(cursor):
    """
    최근 실행한 쿼리의 field names 리스트
    :param cursor:
    :return: list[str]
    """
    return [f[0] for f in cursor.description]


def commit(cursor):
    cursor.connection.commit()


def get_insert_query(table, fields=None, field_count=None):
    """
    format insert query
    :param table: str
    :param fields: list[str]
    :param field_count: int
    :return: str
    """
    if fields:
        q = 'insert into %s ({0}) values ({1});' % table
        l = len(fields)
        q = q.format(','.join(fields), ','.join(['%s'] * l))
    elif field_count:
        q = 'insert into %s values ({0});' % table
        q = q.format(','.join(['%s'] * field_count))
    else:
        raise ValueError('fields or field_count need')

    return q


# noinspection PyShadowingNames
def insert(cursor, table, *args, **field_values):
    """
    db에 레코드 집어넣기
    ex)
    cursor.insert(table, v1, v2,...)
    ex)
    cursor.insert(table, id=v1, word=v2, commit=True)
    :param commit:
    :param cursor:
    :param table:
    :param args:
    :param field_values:
    :return:
    """
    commit = field_values.pop('commit', True)

    q, a = None, None
    if args is not None and len(args) > 0:
        q = get_insert_query(table, field_count=len(args))
        a = args
    elif len(field_values) > 0:
        q = get_insert_query(table, fields=field_values.keys())
        a = field_values.values()
    else:
        raise ValueError('need table, record...')

    cursor.execute(q, args=a)
    if commit:
        cursor.connection.commit()


# noinspection PyShadowingNames
def update(cursor, table, where_kv, commit=True, **field_values):
    """
    db update 쿼리 빌딩 및 실행, 단, commit은
    :param cursor: 커서
    :type cursor: Cursor
    :param table: 테이블 이름
    :type table: str
    :param where_kv: 업데이트 where 조건 dictionary, key:field, value:equal condition only
    :type where_kv: dict
    :param field_values: kwarg 업데이트용
    :type field_values: dict
    :param commit: 커밋 여부
    :type commit: bool
    :return:
    """

    q = """update %s \nset {0} \nwhere {1} """ % table

    fields = field_values.keys()
    kv = ','.join(['{}=%s'.format(f) for f in fields])

    where = ' and '.join(['{}=%s'.format(f) for f in where_kv.keys()])

    q = q.format(kv, where)
    args = field_values.values() + where_kv.values()

    cursor.execute(q, args=args)
    if commit:
        cursor.connection.commit()


# noinspection PyShadowingNames
def insert_or_update(cursor, table, commit=True, **field_values):
    """
    db update 쿼리 빌딩 및 실행, 단, commit은
    :param cursor: 커서
    :type cursor: Cursor
    :param table: 테이블이름
    :type table: str
    :param commit: 커밋 여부
    :type commit: bool
    :param field_values: insert 또는 업데이트 할 필드 및 값 dict pairs
    :type field_values:dict
    :return:
    """

    q = """INSERT INTO %s ({0}) \nVALUES ({1}) \nON DUPLICATE KEY UPDATE {2} """ % table
    l = len(field_values)

    fields = field_values.keys()
    field = ','.join(fields)
    value = ','.join(['%s'] * l)
    kv = ','.join(['{}=%s'.format(f) for f in fields])

    q = q.format(field, value, kv)
    args = field_values.values() * 2

    cursor.execute(q, args=args)
    if commit:
        cursor.connection.commit()

# add property
MySQLdb.cursors.Cursor.fieldcount = property(fieldcount)

# cursor method patching
patch.methods((MySQLdb.cursors.Cursor,), [commit, fieldnames, insert, update, insert_or_update])

