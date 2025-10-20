<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
    <graph role="graph" edgeids="false" edgemode="directed" id="s2">
        <attr name="$version">
            <string>curly</string>
        </attr>
        <node id="n2"/>
        <node id="n1"/>
        <node id="n0"/>
        <node id="n3"/>
        <edge from="n2" to="n0">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n2" to="n2">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n2" to="n2">
            <attr name="label">
                <string>think</string>
            </attr>
        </edge>
        <edge from="n2" to="n3">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n1" to="n1">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n1" to="n0">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n1" to="n3">
            <attr name="label">
                <string>hold</string>
            </attr>
        </edge>
        <edge from="n1" to="n3">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n1" to="n1">
            <attr name="label">
                <string>hasLeft</string>
            </attr>
        </edge>
        <edge from="n0" to="n0">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
        <edge from="n3" to="n3">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
    </graph>
</gxl>
