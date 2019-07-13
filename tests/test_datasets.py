"""
Test dataset functions.
"""

import tempfile
import os

from mira import datasets, core

TEST_VOC_XML_STRING = """
<annotation>
    <folder>VOC2012</folder>
    <filename>test_voc.jpg</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
    </source>
    <size>
        <width>486</width>
        <height>500</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>174</xmin>
            <ymin>101</ymin>
            <xmax>349</xmax>
            <ymax>351</ymax>
        </bndbox>
        <part>
            <name>head</name>
            <bndbox>
                <xmin>169</xmin>
                <ymin>104</ymin>
                <xmax>209</xmax>
                <ymax>146</ymax>
            </bndbox>
        </part>
        <part>
            <name>hand</name>
            <bndbox>
                <xmin>278</xmin>
                <ymin>210</ymin>
                <xmax>297</xmax>
                <ymax>233</ymax>
            </bndbox>
        </part>
        <part>
            <name>foot</name>
            <bndbox>
                <xmin>273</xmin>
                <ymin>333</ymin>
                <xmax>297</xmax>
                <ymax>354</ymax>
            </bndbox>
        </part>
        <part>
            <name>foot</name>
            <bndbox>
                <xmin>319</xmin>
                <ymin>307</ymin>
                <xmax>340</xmax>
                <ymax>326</ymax>
            </bndbox>
        </part>
    </object>
</annotation>
"""


def test_voc():
    """Test VOC loading operations."""
    with tempfile.TemporaryDirectory() as tempdir:
        fpath_xml = os.path.join(tempdir, 'test_voc.xml')
        with open(fpath_xml, 'w') as f:  # pylint: disable=invalid-name
            f.write(TEST_VOC_XML_STRING)
        fpath_jpg = os.path.join(tempdir, 'test_voc.jpg')
        core.Image.new(width=486, height=500, channels=3).save(fpath_jpg)

        original = datasets.load_voc(
            filepaths=[fpath_xml],
            annotation_config=core.AnnotationConfiguration(['person']),
            image_dir=tempdir)
        assert len(original[0].annotations) == 1


def test_via():
    """Test VIA loading and saving operations."""
    collection = datasets.via.load_via('tests/assets/via_project.json')
    with tempfile.TemporaryDirectory() as tempdir:
        savedir = os.path.join(tempdir, 'export')
        datasets.via.save_via(collection, savedir)
        collection_recovered = datasets.via.load_via(
            os.path.join(savedir, 'via.json'))
        assert len(collection_recovered) == 1
        assert len(collection_recovered[0].annotations) == 1
        bbox1 = collection[0].annotations[0].selection.bbox()
        bbox2 = collection_recovered[0].annotations[0].selection.bbox()
        print(bbox1, bbox2)
        assert all(v1 == v2 for v1, v2 in zip(bbox1, bbox2))
