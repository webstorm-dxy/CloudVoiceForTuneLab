using TuneLab.CloudVoice;
using TuneLab.Foundation;
using TuneLab.SDK;
using Xunit;

namespace TuneLab.CloudVoice.Tests;

public class SettingsTests
{
    [Fact]
    public void Defaults_AreEmptyAndOffline()
    {
        var settings = CloudVoiceSettings.FromPropertyObject(PropertyObject.Empty);

        Assert.Equal("", settings.Endpoint);
        Assert.Equal("", settings.ApiKey);
        Assert.Equal(48000, settings.SampleRate);
        Assert.False(settings.HasCloudEndpoint);
    }

    [Fact]
    public void ReadsOnlyEndpointAndApiKey_AndTrimsEndpoint()
    {
        var settings = CloudVoiceSettings.FromPropertyObject(new PropertyObject(new Map<string, PropertyValue>
        {
            ["endpoint"] = "  https://cloud.test/v1/synthesize  ",
            ["api_key"] = "sk-1",
        }));

        Assert.Equal("https://cloud.test/v1/synthesize", settings.Endpoint);
        Assert.Equal("sk-1", settings.ApiKey);
        Assert.True(settings.HasCloudEndpoint);
    }

    [Fact]
    public void LegacyFields_AreIgnored()
    {
        var settings = CloudVoiceSettings.FromPropertyObject(new PropertyObject(new Map<string, PropertyValue>
        {
            ["provider"] = "http",
            ["voices_json"] = "[{\"id\":\"x\"}]",
            ["model"] = "x",
        }));

        Assert.Equal("", settings.Endpoint);
        Assert.False(settings.HasCloudEndpoint);
    }

    [Fact]
    public void SettingsConfig_HasOnlyServiceUrlAndApiKey()
    {
        var config = CloudVoiceSettings.GetSettingsConfig(PropertyObject.Empty, "en-US");
        var keys = config.Properties.Keys.Select(key => key.Id).ToList();

        Assert.Equal(2, keys.Count);
        Assert.Contains("endpoint", keys);
        Assert.Contains("api_key", keys);
        var apiKey = (TextBoxConfig)config.Properties.Values[keys.IndexOf("api_key")];
        Assert.True(apiKey.IsPassword);
    }

    [Theory]
    [InlineData("http://host/synthesize", "http://host/voices")]
    [InlineData("http://host:1234/synthesize", "http://host:1234/voices")]
    [InlineData("https://cloud.test/v1/synthesize", "https://cloud.test/v1/voices")]
    [InlineData("https://cloud.test/v1/", "https://cloud.test/v1/voices")]
    public void ResolveVoicesUrl_ReplacesLastSegment(string endpoint, string expected)
    {
        var settings = new CloudVoiceSettings { Endpoint = endpoint };

        Assert.Equal(expected, settings.ResolveVoicesUrl());
    }
}

public class CatalogTests
{
    const string ProviderJson =
        "{\"voices\":[" +
        "{\"id\":\"soprano\",\"name\":\"Soprano\",\"description\":\"d1\",\"voice\":\"prov-soprano\"}," +
        "{\"id\":\"alto\"}]}";

    [Fact]
    public void OfflineDemo_ExposesMockVoices()
    {
        var catalog = CloudVoiceCatalogBuilder.BuildOfflineDemo("", "en-US");

        Assert.Equal(3, catalog.Infos.Count);
        Assert.Single(catalog.Layout);
        var group = Assert.IsType<VoiceSourceLayoutGroup>(catalog.Layout[0]);
        Assert.Equal(3, group.Items.Count);
        Assert.Equal(CloudVoiceSettings.ProviderMock, catalog.Routes["mock-soprano"].ProviderKind);
    }

    [Fact]
    public void ProviderCatalog_ParsesVoicesObjectAndRoutes()
    {
        var result = CloudVoiceCatalogBuilder.ParseProviderCatalog(ProviderJson, "", "en-US");

        Assert.True(result.Success, result.Error);
        var catalog = result.Catalog!;
        Assert.Equal(2, catalog.Infos.Count);
        Assert.Equal("Soprano", catalog.Infos["soprano"].Name);
        Assert.Equal("d1", catalog.Infos["soprano"].Description);
        Assert.Equal("prov-soprano", catalog.Routes["soprano"].ProviderVoiceId);
        Assert.Equal("alto", catalog.Routes["alto"].ProviderVoiceId);
        Assert.Equal(CloudVoiceSettings.ProviderHttp, catalog.Routes["soprano"].ProviderKind);
        Assert.Single(catalog.Layout);
    }

    [Fact]
    public void ProviderCatalog_AcceptsBareArray()
    {
        var result = CloudVoiceCatalogBuilder.ParseProviderCatalog(
            "[{\"id\":\"a\",\"name\":\"A\"}]", "", "zh-CN");

        Assert.True(result.Success, result.Error);
        Assert.Single(result.Catalog!.Infos);
    }

    [Theory]
    [InlineData("{not json")]
    [InlineData("{\"items\":[]}")]
    [InlineData("{\"voices\":[]}")]
    public void ProviderCatalog_RejectsInvalidOrEmpty(string json)
    {
        var result = CloudVoiceCatalogBuilder.ParseProviderCatalog(json, "", "en-US");

        Assert.False(result.Success);
        Assert.NotNull(result.Error);
    }

    [Fact]
    public void ProviderCatalog_DeduplicatesIds()
    {
        var result = CloudVoiceCatalogBuilder.ParseProviderCatalog(
            "{\"voices\":[{\"id\":\"a\"},{\"id\":\"a\",\"name\":\"dup\"}]}", "", "en-US");

        Assert.True(result.Success, result.Error);
        Assert.Single(result.Catalog!.Infos);
        Assert.Equal("a", result.Catalog.Infos["a"].Name);
    }
}