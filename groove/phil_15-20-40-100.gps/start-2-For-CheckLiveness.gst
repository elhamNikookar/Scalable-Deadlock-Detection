<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<gxl xmlns="http://www.gupro.de/GXL/gxl-1.0.dtd">
    <graph role="graph" edgeids="false" edgemode="directed" id="start-2-For-CheckLiveness">
        <attr name="transitionLabel">
            <string></string>
        </attr>
        <attr name="enabled">
            <string>true</string>
        </attr>
        <attr name="priority">
            <string>0</string>
        </attr>
        <attr name="printFormat">
            <string></string>
        </attr>
        <attr name="remark">
            <string></string>
        </attr>
        <attr name="$version">
            <string>curly</string>
        </attr>
        <node id="n275">
            <attr name="layout">
                <string>557 227 32 24</string>
            </attr>
        </node>
        <node id="n277">
            <attr name="layout">
                <string>421 362 38 48</string>
            </attr>
        </node>
        <node id="n282">
            <attr name="layout">
                <string>386 94 38 48</string>
            </attr>
        </node>
        <node id="n285">
            <attr name="layout">
                <string>263 254 32 24</string>
            </attr>
        </node>
        <edge from="n275" to="n275">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
        <edge from="n277" to="n277">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n277" to="n277">
            <attr name="label">
                <string>think</string>
            </attr>
        </edge>
        <edge from="n277" to="n275">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n277" to="n285">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n282" to="n282">
            <attr name="label">
                <string>Phil</string>
            </attr>
        </edge>
        <edge from="n282" to="n282">
            <attr name="label">
                <string>hungry</string>
            </attr>
        </edge>
        <edge from="n282" to="n282">
            <attr name="label">
                <string>specific</string>
            </attr>
        </edge>
        <edge from="n282" to="n285">
            <attr name="label">
                <string>right</string>
            </attr>
        </edge>
        <edge from="n282" to="n275">
            <attr name="label">
                <string>left</string>
            </attr>
        </edge>
        <edge from="n285" to="n285">
            <attr name="label">
                <string>Fork</string>
            </attr>
        </edge>
    </graph>
</gxl>
