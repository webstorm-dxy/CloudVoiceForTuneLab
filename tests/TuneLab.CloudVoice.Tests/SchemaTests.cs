using TuneLab.CloudVoice;
using TuneLab.SDK;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class SchemaTests
{
    const string FullSchema = """
    {
      "pitch": { "auto": true },
      "partProperties": [
        { "id": "PitchMode", "label": "Pitch mode", "type": "combo", "default": "predict",
          "options": [ { "value": "tunelab", "label": "TuneLab" }, { "value": "predict", "label": "Auto" } ] },
        { "id": "amount", "label": "Amount", "type": "slider", "min": 0, "max": 1, "default": 0.5 }
      ],
      "noteProperties": [
        { "id": "language", "label": "Language", "type": "combo", "default": "",
          "options": [ { "value": "", "label": "Follow part" }, { "value": "mandarin", "label": "Mandarin" } ] },
        { "id": "shift", "type": "number", "min": -12, "max": 12, "default": 0 }
      ],
      "automations": [
        { "id": "DYN", "label": "Dynamics", "min": -1, "max": 1, "default": 0, "color": "#73ACE5" },
        { "id": "PIT", "label": "Pitch bend", "min": -1, "max": 1, "default": 0 },
        { "id": "segment", "label": "Segment", "min": 0, "max": 1 }
      ],
      "synthesizedParameters": [
        { "id": "energy", "label": "Energy", "min": 0, "max": 100, "color": "#8ADF73" }
      ]
    }
    """;

    [Fact]
    public void Parse_ReadsPitchModesPropertiesAutomationsAndEchoTracks()
    {
        var schema = CloudVoiceSchema.Parse(FullSchema);

        Assert.NotNull(schema);
        Assert.True(schema!.PitchAuto);
        Assert.Equal(2, schema.PartProperties.Count);
        Assert.Equal(2, schema.NoteProperties.Count);
        Assert.Equal(3, schema.Automations.Count);
        Assert.Single(schema.SynthesizedParameters);

        var pitchMode = schema.PartProperties[0];
        Assert.Equal("PitchMode", pitchMode.Id);
        Assert.Equal(CloudPropertyKind.Combo, pitchMode.Kind);
        Assert.Equal("predict", pitchMode.TextDefault);
        Assert.Equal(2, pitchMode.Options.Count);
        Assert.Equal("tunelab", pitchMode.Options[0].Value);

        Assert.Equal(CloudPropertyKind.Slider, schema.PartProperties[1].Kind);
        Assert.Equal(0.5, schema.PartProperties[1].NumberDefault);

        Assert.Equal(CloudPropertyKind.Number, schema.NoteProperties[1].Kind);
        Assert.Equal(0.0, schema.Automations[0].Default);
        Assert.Null(schema.Automations[2].Default);
    }

    [Fact]
    public void BuildConfigs_ExposeSchemaDeclarationsToTheHost()
    {
        var schema = CloudVoiceSchema.Parse(FullSchema)!;

        var partConfig = schema.BuildPartPropertyConfig();
        Assert.Equal(2, partConfig.Properties.Count);
        Assert.Equal(new[] { "PitchMode", "amount" }, partConfig.Properties.Keys.Select(key => key.Id).ToArray());

        var noteConfig = schema.BuildNotePropertyConfig();
        Assert.Equal(2, noteConfig.Properties.Count);

        var automations = schema.BuildAutomationConfigs();
        Assert.Equal(3, automations.Count);
        Assert.Equal(new[] { "DYN", "PIT", "segment" }, automations.Keys.Select(key => key.Id).ToArray());

        var echo = schema.BuildSynthesizedParameterConfigs();
        Assert.Single(echo);
        Assert.Equal("energy", echo.Keys[0].Id);
    }

    [Fact]
    public void Parse_EmptyObject_ReturnsEmptySchemaWithManualPitch()
    {
        var schema = CloudVoiceSchema.Parse("{}");

        Assert.NotNull(schema);
        Assert.False(schema!.PitchAuto);
        Assert.Empty(schema.PartProperties);
        Assert.Empty(schema.Automations);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("{not json")]
    public void Parse_InvalidJson_ReturnsNull(string json)
    {
        Assert.Null(CloudVoiceSchema.Parse(json));
    }

    [Fact]
    public void Defaults_KeepLegacyDeclarationsForOldServers()
    {
        var schema = CloudVoiceSchema.Defaults;

        Assert.False(schema.PitchAuto);
        Assert.Equal(new[] { "dynamics", "breathiness" }, schema.NotePropertyKeys.ToArray());
        Assert.Equal(new[] { "dynamics", "breathiness" }, schema.AutomationKeys.ToArray());
        Assert.Equal(new[] { "energy" }, schema.SynthesizedParameterKeys.ToArray());
        Assert.Empty(schema.PartProperties);
    }
}