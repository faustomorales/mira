from mira import datasets, core

test_string = """
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
    with open('tests/assets/test_voc.xml', 'w') as f:
        f.write(test_string)
    core.Image.new(
        width=486,
        height=500,
        channels=3
    ).save(
        'tests/assets/test_voc.jpg'
    )

    original = datasets.load_voc(
        filepaths=['tests/assets/test_voc.xml'],
        annotation_config=core.AnnotationConfiguration(['person']),
        image_dir='tests/assets/'
    )
    assert len(original[0].annotations) == 1