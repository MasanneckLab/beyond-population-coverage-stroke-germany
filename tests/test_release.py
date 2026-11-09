from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geostroke.site_contacts import contact
from scripts.check_public_release import audit_file
from scripts.compute_str_travel_time_sparse_sensitivity_2024 import gravity_alloc_sparse
from scripts import update_publication_figure_labels_borders as plot
from scripts.compute_str_travel_time_sensitivity_2024 import post_matrix
from geostroke.iso_manager import _compute_isochrones


class PublicationTests(unittest.TestCase):
    def test_matrix_direction_and_units(self):
        from unittest.mock import Mock
        response = Mock(ok=True)
        response.json.return_value = {'durations':[[600,None]], 'distances':[[7.2,None]]}
        with patch('scripts.compute_str_travel_time_sensitivity_2024.requests.post', return_value=response) as request:
            minutes, km = post_matrix('http://example.invalid', [[7,50]], [[8,51],[9,52]])
        payload = request.call_args.kwargs['json']
        self.assertEqual(payload['sources'], [0])
        self.assertEqual(payload['destinations'], [1,2])
        self.assertEqual(payload['units'], 'km')
        self.assertEqual(minutes[0,0],10)
        self.assertEqual(km[0,0],7.2)
        self.assertTrue(np.isnan(minutes[0,1]))

    def test_isochrone_request_batches_keep_all_twelve_ranges(self):
        from unittest.mock import Mock
        payloads=[]
        def post(*args, **kwargs):
            payloads.append(kwargs['json'])
            response = Mock()
            response.json.return_value = {'features':[{'geometry':{'type':'Polygon',
                'coordinates':[[[0,0],[1,0],[1,1],[0,0]]]}, 'properties':{'value':value}}
                for value in kwargs['json']['range']]}
            return response
        with patch('geostroke.iso_manager.requests.post', side_effect=post):
            result = _compute_isochrones(7,50)
        self.assertEqual(len(result),12)
        self.assertEqual([len(p['range']) for p in payloads],[10,2])
        self.assertEqual([v for p in payloads for v in p['range']],list(range(300,3601,300)))

    def test_non_overlapping_boundary_bins(self):
        for therapy, times, bounds in [('IVT',[0,9.9,10,19.9,20,29.9,30,45],[0,10,20,30,np.inf]),
                ('MT',[0,14.9,15,29.9,30,44.9,45,90],[0,15,30,45,np.inf])]:
            frame = pd.DataFrame(dict(time=times, STR=range(1,9), urban_rural=['urban']*8))
            fig, ax = plt.subplots()
            plot._scatter_decay(ax,frame,therapy,'STR','time','Title','STR',bounds,'Minutes')
            line = next(x for x in ax.lines if x.get_label() == 'Binned mean')
            self.assertEqual(list(line.get_ydata()), [1.5,3.5,5.5,7.5])
            plt.close(fig)

    def test_comparator_annotation_recomputed(self):
        frame = pd.DataFrame(dict(urban_rural=['urban','rural']*4,
            STR_IVT=range(1,9),STR_MT=range(1,9),STR_IVT_time_primary=range(8,0,-1),
            STR_MT_time_primary=range(8,0,-1), nearest_IVT_time_min=[0,9.9,10,19.9,20,29.9,30,45],
            nearest_MT_time_min=[0,14.9,15,29.9,30,44.9,45,60]))
        titles=[]
        with TemporaryDirectory() as folder:
            frame.to_csv(Path(folder)/'county_STR_travel_time_sensitivity_2024.csv',index=False)
            with patch.object(plot,'AUDEBERT_DIR',Path(folder)), patch.object(plot,'save_existing_formats',
                    lambda fig,*args: titles.extend(ax.get_title() for ax in fig.axes)):
                plot.make_s5_scatter_only()
        self.assertIn('rho=-1.00',titles[0])
        self.assertIn('rho=-1.00',titles[1])

    def test_sparse_allocation_conserves_counts_with_zero_and_cutoff_fallback(self):
        hospitals = pd.DataFrame(dict(total_cases=[20,30]))
        nodes = pd.DataFrame(dict(node_pop=[100,200,300]))
        pairs = pd.DataFrame(dict(facility_idx=[0,0,1,1], node_idx=[0,1,1,2], duration_min=[0,10,100,200]))
        for cutoff in [60,90,None]:
            result = gravity_alloc_sparse(hospitals,nodes,pairs,2,cutoff)
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(result.sum(),50)

    def test_site_contact_precedes_group_headquarters(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'1-2-2024-xml.xml'
            path.write_text('<Report><Krankenhauskontaktdaten><Name>Group</Name></Krankenhauskontaktdaten>'
                '<Standortkontaktdaten><Standortnummer>2</Standortnummer><Name>Site</Name>'
                '<Kontakt_Zugang><Strasse>Campus</Strasse></Kontakt_Zugang></Standortkontaktdaten></Report>')
            self.assertEqual(contact(path)['street'],'Campus')
            path.write_text('<Report><Standortkontaktdaten/><Standortkontaktdaten/></Report>')
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                contact(path)

    def test_publication_documents_rejected(self):
        with self.assertRaises(ValueError):
            audit_file('ESM_1.pdf', b'PDF')

    def test_template_rows_and_private_cache_rejected(self):
        audit_file('data/templates/example.csv', b'name,latitude,longitude\n')
        with self.assertRaises(ValueError):
            audit_file('data/templates/example.csv', b'name,latitude,longitude\nExample,50,7\n')
        with self.assertRaises(ValueError):
            audit_file('data/private/routes.csv', b'origin,destination\n')

    def test_hospital_columns_rejected_in_aggregate(self):
        with self.assertRaises(ValueError):
            audit_file('data/aggregate/county_STR_analysis.csv', b'facility_id,latitude\n')


if __name__ == '__main__':
    unittest.main()
