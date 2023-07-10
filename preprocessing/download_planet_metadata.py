import argparse
import json
import os
import pickle
import time
from queue import Queue

import requests
from requests import Request

item_type = 'PSScene'
# os.environ['PLANET_API_KEY'] = 'REDACTED'


def ensure_request(sess, url, wait=.1, exp=2, max_wait=5, max_retry=10, stream=False, method='GET'):
    retry = 0
    code = 0
    time.sleep(wait)
    while True:
        # print('connecting...')
        try:
            if stream:
                req = Request(
                    method,
                    url,

                ).prepare()
                response = sess.send(
                    req,
                    stream=stream
                )
            else:
                response = sess.get(url)
            code = response.status_code
            if response.status_code // 100 == 2:
                return response
        except Exception as e:
            print(e)
        finally:
            # print(response.status_code)
            time.sleep(wait)
            if wait < max_wait:
                wait *= exp
            retry += 1
            if retry > max_retry:
                print(f'Request failed: {url}, {code}')
                return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson_file', type=str)
    parser.add_argument('--output_id_file', type=str)
    parser.add_argument('--output_pickle_file', type=str)
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    polling_queue = Queue()
    items_collection = {}
    ids = []
    session = requests.Session()
    session.auth = (os.environ['PLANET_API_KEY'], '')
    if args.geojson_file:
        geo_json_geometry = json.load(open(args.geojson_file))['geometry']


    # this large search filter produces all PlanetScope imagery for 1 day
    def search(day):
        very_large_search = {
            "name": "very_large_search",
            "item_types": ["PSScene"],
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {
                            "gte": f"{args.year:04}-{args.month:02}-{day:02}T00:00:00Z",
                            "lte": f"{args.year:04}-{args.month:02}-{(day + 1):02}T00:00:00Z"
                        }
                    },
                    {
                        "type": "AssetFilter",
                        "config": [
                            "ortho_analytic_4b",
                            "ortho_analytic_4b_xml",
                            "ortho_udm2"
                        ]
                    },
                ]
            }
        }
        if args.geojson_file:
            very_large_search['filter']['config'].append({
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": geo_json_geometry
            })
        print('submitting search')
        saved_search = \
            session.post(
                'https://api.planet.com/data/v1/searches/',
                json=very_large_search)
        # after you create a search, save the id. This is what is needed
        # to execute the search.
        print(saved_search.status_code, saved_search.text)
        saved_search_id = saved_search.json()["id"]

        print('fetching page')
        first_page = \
            ("https://api.planet.com/data/v1/searches/{}" +
             "/results?_page_size={}").format(saved_search_id, 200)
        EOF = False
        current_page = first_page
        page_count = 1
        while not EOF:
            print(f'fetching page {page_count}', flush=True)
            response = ensure_request(session, current_page)
            page = response.json()
            for item in page['features']:
                ids.append(item['id'] + '\n')
                items_collection[item['id']] = {
                    'properties': item['properties'],
                    'geometry': item['geometry']
                }
            page_count += 1
            current_page = page["_links"].get("_next")
            if not current_page:
                EOF = True


    for day in range(1, args.days):
        search(day)
        print(day)
    if args.output_id_file:
        with open(args.output_id_file, 'w+') as f:
            f.writelines(ids)
    if args.output_pickle_file:
        pickle.dump(items_collection, open(args.output_pickle_file, 'wb+'))
